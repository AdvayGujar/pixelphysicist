"""M7: safe AST grammar + evaluation. (Day 3.)

A model term is a scalar expression in ``q`` and ``v`` only, e.g.
``abs(v)*v`` or ``sin(q)``. Terms are validated with an ``ast`` whitelist
and evaluated through a compiled code object with an empty-builtins,
NumPy-only namespace -- a raw model string is never passed to ``eval``.
"""
from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Any

import numpy as np

MAX_TERM_LEN = 64
MAX_TERMS = 8

ALLOWED_NAMES = ("q", "v")
ALLOWED_FUNCS: dict[str, Any] = {
    "sin": np.sin,
    "cos": np.cos,
    "tanh": np.tanh,
    "sign": np.sign,
    "abs": np.abs,
    "sqrt": np.sqrt,
    "exp": np.exp,
}


class _TermValidator(ast.NodeVisitor):
    """Accept exactly the whitelisted grammar; anything else raises."""

    def generic_visit(self, node: ast.AST) -> None:  # type: ignore[override]
        raise ValueError(f"forbidden syntax: {type(node).__name__}")

    def visit_Expression(self, node: ast.Expression) -> None:
        self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            raise ValueError(f"forbidden operator: {type(node.op).__name__}")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ValueError(f"forbidden operator: {type(node.op).__name__}")
        self.visit(node.operand)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if not isinstance(func, ast.Name) or func.id not in ALLOWED_FUNCS:
            raise ValueError("only whitelisted single-arg functions may be called")
        if len(node.args) != 1 or node.keywords:
            raise ValueError("functions take exactly one positional argument")
        self.visit(node.args[0])

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in ALLOWED_NAMES:
            raise ValueError(f"unknown name: {node.id!r}")

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"only numeric literals allowed, got {node.value!r}")


def validate_term(term: str) -> str:
    """Check one term against the grammar; return its stripped form."""
    if not isinstance(term, str) or not term.strip():
        raise ValueError("term must be a non-empty string")
    if len(term) > MAX_TERM_LEN:
        raise ValueError(f"term longer than {MAX_TERM_LEN} chars: {term!r}")
    try:
        tree = ast.parse(term, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid term {term!r}: {exc}") from exc
    _TermValidator().visit(tree)
    return term.strip()


def validate_terms(terms: list[str]) -> list[str]:
    """Check a term library: 1..MAX_TERMS valid terms."""
    if not 1 <= len(terms) <= MAX_TERMS:
        raise ValueError(f"need 1..{MAX_TERMS} terms, got {len(terms)}")
    return [validate_term(t) for t in terms]


def compile_term(term: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Validate a term and return ``f(q, v)`` over NumPy arrays."""
    clean = validate_term(term)
    code = compile(ast.parse(clean, mode="eval"), "<term>", "eval")

    def evaluate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        namespace: dict[str, Any] = dict(ALLOWED_FUNCS)
        namespace.update({"q": q, "v": v})
        # Terms like 1/q hit zero crossings routinely; callers check
        # finiteness (weak_init rejects, the shooter routes around). Warn
        # nowhere: a warning per integrator step would flood Day-5 logs.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            result = eval(code, {"__builtins__": {}}, namespace)  # noqa: S307
        values = np.asarray(result, dtype=float)
        if values.ndim == 0:
            return np.full(np.shape(q), float(values))
        if values.shape != np.shape(q):
            raise ValueError(f"term {term!r} produced shape {values.shape}")
        return values

    return evaluate


def basis_matrix(terms: list[str], q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Evaluate every term on ``(q, v)``; columns of the design matrix."""
    clean = validate_terms(terms)
    q_arr = np.asarray(q, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    return np.column_stack([compile_term(t)(q_arr, v_arr) for t in clean])
