#!/usr/bin/env python3
"""Parse SISSO models, evaluate expressions, and refit coefficients robustly."""

from __future__ import annotations

import csv
import math
import re
import statistics
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


def number(text: str) -> float:
    return float(text.replace("D", "E").replace("d", "e"))


def metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    if len(actual) != len(predicted) or not actual:
        raise ValueError("Metrics require equally sized, non-empty vectors")
    residuals = [pred - true for true, pred in zip(actual, predicted)]
    mean_actual = statistics.fmean(actual)
    sse = sum(value * value for value in residuals)
    sst = sum((value - mean_actual) ** 2 for value in actual)
    return {
        "R2": 1.0 - sse / sst if sst > 0 else float("nan"),
        "RMSE": math.sqrt(sse / len(actual)),
        "MAE": statistics.fmean(abs(value) for value in residuals),
        "MaxAE": max(abs(value) for value in residuals),
    }


@dataclass
class Node:
    kind: str
    value: str | None = None
    left: "Node | None" = None
    right: "Node | None" = None

    def render(self) -> str:
        if self.kind == "feature":
            return str(self.value)
        if self.kind == "neg":
            return f"(-{self.left.render()})"
        if self.kind in {"abs", "sqrt", "exp"}:
            return f"{self.kind}({self.left.render()})"
        if self.kind == "pow":
            return f"({self.left.render()})^{self.value}"
        return f"({self.left.render()}{self.value}{self.right.render()})"

    def evaluate_one(self, values: dict[str, float]) -> float:
        if self.kind == "feature":
            result = values[str(self.value)]
        elif self.kind == "neg":
            result = -self.left.evaluate_one(values)
        elif self.kind == "abs":
            result = abs(self.left.evaluate_one(values))
        elif self.kind == "sqrt":
            result = math.sqrt(self.left.evaluate_one(values))
        elif self.kind == "exp":
            result = math.exp(self.left.evaluate_one(values))
        elif self.kind == "pow":
            result = self.left.evaluate_one(values) ** int(str(self.value))
        else:
            lhs = self.left.evaluate_one(values)
            rhs = self.right.evaluate_one(values)
            if self.value == "+":
                result = lhs + rhs
            elif self.value == "-":
                result = lhs - rhs
            elif self.value == "/":
                result = lhs / rhs
            else:
                raise ValueError(f"Unsupported operator: {self.value}")
        if not math.isfinite(result):
            raise ValueError(f"Non-finite expression value for {self.render()}")
        return result

    def evaluate(self, rows: list[dict[str, float]]) -> list[float]:
        return [self.evaluate_one(row) for row in rows]

    def denominator_nodes(self) -> list["Node"]:
        found: list[Node] = []
        if self.kind == "binary":
            found.extend(self.left.denominator_nodes())
            found.extend(self.right.denominator_nodes())
            if self.value == "/":
                found.append(self.right)
        elif self.kind == "pow":
            found.extend(self.left.denominator_nodes())
            if str(self.value) == "-1":
                found.append(self.left)
        elif self.left is not None:
            found.extend(self.left.denominator_nodes())
        return found


class ExpressionParser:
    token_re = re.compile(r"feature\d+|sqrt|abs|exp|\^|-?\d+|[()+\-/]")

    def __init__(self, text: str):
        compact = re.sub(r"\s+", "", text)
        self.tokens = self.token_re.findall(compact)
        if "".join(self.tokens) != compact:
            raise ValueError(f"Could not tokenize expression: {text}")
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self, expected: str | None = None) -> str:
        token = self.peek()
        if token is None or (expected is not None and token != expected):
            raise ValueError(f"Expected {expected!r}, got {token!r}")
        self.pos += 1
        return token

    def parse(self) -> Node:
        node = self.parse_sum()
        if self.peek() is not None:
            raise ValueError(f"Unexpected token {self.peek()!r}")
        return node

    def parse_sum(self) -> Node:
        node = self.parse_term()
        while self.peek() in {"+", "-"}:
            operator = self.take()
            node = Node("binary", operator, node, self.parse_term())
        return node

    def parse_term(self) -> Node:
        node = self.parse_power()
        while self.peek() == "/":
            self.take("/")
            node = Node("binary", "/", node, self.parse_power())
        return node

    def parse_power(self) -> Node:
        node = self.parse_unary()
        if self.peek() == "^":
            self.take("^")
            exponent = self.take()
            if exponent not in {"2", "-1"}:
                raise ValueError(f"Unsupported exponent: {exponent}")
            node = Node("pow", exponent, node)
        return node

    def parse_unary(self) -> Node:
        if self.peek() == "-":
            self.take("-")
            return Node("neg", left=self.parse_unary())
        if self.peek() in {"sqrt", "abs", "exp"}:
            function = self.take()
            self.take("(")
            child = self.parse_sum()
            self.take(")")
            return Node(function, left=child)
        if self.peek() == "(":
            self.take("(")
            node = self.parse_sum()
            self.take(")")
            return node
        token = self.take()
        if not token.startswith("feature"):
            raise ValueError(f"Expected feature, got {token}")
        return Node("feature", token)


def parse_models(text: str) -> list[dict[str, object]]:
    header = re.compile(r"\b(\d+)D descriptor/model\(y=sum\(ci\*di\)\+c0\)\s*:")
    starts = list(header.finditer(text))
    models: list[dict[str, object]] = []
    for index, match in enumerate(starts):
        dimension = int(match.group(1))
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[match.end():end]
        expressions = [
            item.group(2).strip()
            for item in re.finditer(r"d(\d+)\s*=\s*(.*?)\s+feature_ID:", block)
        ]
        coefficient_match = re.search(r"coeff\.\(ci\):\s*(.*)", block)
        intercept_match = re.search(r"\bc0:\s*(%s)" % FLOAT_RE, block)
        reported_match = re.search(
            r"RMSE and MaxAE:\s*(%s)\s+(%s)" % (FLOAT_RE, FLOAT_RE), block
        )
        if len(expressions) != dimension or not coefficient_match or not intercept_match:
            continue
        coefficients = [number(value) for value in re.findall(FLOAT_RE, coefficient_match.group(1))]
        if len(coefficients) != dimension:
            continue
        models.append({
            "dimension": dimension,
            "expressions": expressions,
            "sisso_coefficients": coefficients,
            "sisso_intercept": number(intercept_match.group(1)),
            "reported_rmse": number(reported_match.group(1)) if reported_match else float("nan"),
            "reported_maxae": number(reported_match.group(2)) if reported_match else float("nan"),
        })
    return models


def _decimal_solve(matrix: list[list[Decimal]], rhs: list[Decimal]) -> tuple[list[Decimal], float]:
    n = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    pivot_abs: list[Decimal] = []
    with localcontext() as context:
        context.prec = 80
        for column in range(n):
            pivot = max(range(column, n), key=lambda row: abs(a[row][column]))
            if a[pivot][column] == 0:
                raise ArithmeticError("Rank-deficient composite-descriptor matrix")
            if pivot != column:
                a[column], a[pivot] = a[pivot], a[column]
            pivot_value = a[column][column]
            pivot_abs.append(abs(pivot_value))
            for item in range(column, n + 1):
                a[column][item] /= pivot_value
            for row in range(n):
                if row == column:
                    continue
                factor = a[row][column]
                if factor == 0:
                    continue
                for item in range(column, n + 1):
                    a[row][item] -= factor * a[column][item]
        solution = [a[row][n] for row in range(n)]
    smallest = min(pivot_abs)
    pivot_ratio = float(max(pivot_abs) / smallest) if smallest != 0 else float("inf")
    return solution, pivot_ratio


def refit_same_expression(
    expressions: list[str], feature_rows: list[dict[str, float]], targets: list[float]
) -> dict[str, object]:
    """OLS refit of a selected SISSO expression using 80-digit normal equations.

    SISSO output prints a limited number of coefficient digits. Re-fitting the same
    selected descriptors removes prediction artifacts caused only by that printing
    precision; it does not alter screening or expression selection.
    """
    nodes = [ExpressionParser(expression).parse() for expression in expressions]
    descriptors = [[node.evaluate_one(row) for node in nodes] for row in feature_rows]
    dimension = len(nodes)
    means = [statistics.fmean(row[j] for row in descriptors) for j in range(dimension)]
    scales = []
    for j in range(dimension):
        variance = statistics.fmean((row[j] - means[j]) ** 2 for row in descriptors)
        scale = math.sqrt(variance)
        if not math.isfinite(scale) or scale == 0:
            raise ArithmeticError(f"Constant composite descriptor at D={dimension}, column {j + 1}")
        scales.append(scale)
    standardized = [
        [1.0, *[(row[j] - means[j]) / scales[j] for j in range(dimension)]]
        for row in descriptors
    ]
    with localcontext() as context:
        context.prec = 80
        x = [[Decimal(str(value)) for value in row] for row in standardized]
        y = [Decimal(str(value)) for value in targets]
        size = dimension + 1
        gram = [[sum(row[i] * row[j] for row in x) for j in range(size)] for i in range(size)]
        cross = [sum(row[i] * value for row, value in zip(x, y)) for i in range(size)]
        standardized_coefficients, pivot_ratio = _decimal_solve(gram, cross)
        gamma = [float(value) for value in standardized_coefficients[1:]]
        coefficients = [gamma[j] / scales[j] for j in range(dimension)]
        intercept = float(standardized_coefficients[0]) - sum(
            coefficients[j] * means[j] for j in range(dimension)
        )
    predictions = [
        intercept + sum(coef * value for coef, value in zip(coefficients, row))
        for row in descriptors
    ]
    if not all(math.isfinite(value) for value in predictions):
        raise ArithmeticError("Non-finite OLS reconstruction")
    return {
        "expressions": expressions,
        "nodes": nodes,
        "coefficients": coefficients,
        "intercept": intercept,
        "training_predictions": predictions,
        "training_metrics": metrics(targets, predictions),
        "normal_equation_pivot_ratio": pivot_ratio,
    }


def read_desc_file(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        next(handle)
        for line in handle:
            fields = line.split()
            if fields:
                rows.append([number(value) for value in fields])
    return rows


def normalize_expression(text: str) -> str:
    return re.sub(r"\s+", "", text)


def mapped_expression(expression: str, mapping: dict[str, str]) -> str:
    return re.sub(r"feature\d+", lambda match: mapping.get(match.group(0), match.group(0)), expression)

