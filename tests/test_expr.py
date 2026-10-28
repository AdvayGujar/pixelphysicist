"""M13: expr grammar tests -- valid terms, rejections, injection attempts."""

import numpy as np
import pytest

from pixelphys import expr


def test_baseline_library_validates() -> None:
    from pixelphys.physics import BASELINE_TERMS

    assert expr.validate_terms(list(BASELINE_TERMS)) == list(BASELINE_TERMS)


def test_menu_terms_validate() -> None:
    menu = ["abs(v)*v", "sign(v)", "q*v", "v**3", "sin(q)*v", "cos(q)*v"]
    assert expr.validate_terms(menu) == menu


def test_evaluation_matches_numpy() -> None:
    q = np.array([0.0, 0.5, -1.0])
    v = np.array([1.0, -2.0, 0.5])
    assert np.allclose(expr.compile_term("abs(v)*v")(q, v), np.abs(v) * v)
    assert np.allclose(expr.compile_term("q*v + sin(q)")(q, v), q * v + np.sin(q))
    assert np.allclose(expr.compile_term("1")(q, v), np.ones(3))


def test_basis_matrix_shape_and_constant_column() -> None:
    q = np.linspace(0.0, 1.0, 11)
    v = np.linspace(1.0, 2.0, 11)
    mat = expr.basis_matrix(["1", "q", "abs(v)*v"], q, v)
    assert mat.shape == (11, 3)
    assert np.allclose(mat[:, 0], 1.0)
    assert np.allclose(mat[:, 1], q)
    assert np.allclose(mat[:, 2], np.abs(v) * v)


@pytest.mark.parametrize("bad", [
    "os.system('x')",           # attribute call
    "__import__('os')",         # dunder import
    "q.__class__",              # dunder access
    "q[0]",                     # subscript
    "q > 0",                    # comparison
    "q and v",                  # bool op
    "q if q else v",            # if expression
    "lambda q: q",              # lambda
    "[q for q in v]",           # comprehension
    "eval('1')",                # non-whitelisted call
    "np.sin(q)",                # module attribute
    "sin(q, v)",                # wrong arity
    "sin(q=1)",                 # keyword arg
    "w",                        # unknown name
    "sin",                      # bare function name
    "'x'",                      # string literal
    "True",                     # bool literal
    "",                         # empty
    "q + ",                     # syntax error
])
def test_invalid_terms_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        expr.validate_term(bad)


def test_term_length_capped() -> None:
    with pytest.raises(ValueError, match="longer than"):
        expr.validate_term("q+" * 40)


def test_term_count_capped() -> None:
    with pytest.raises(ValueError, match="1..8"):
        expr.validate_terms(["q"] * 9)
    with pytest.raises(ValueError, match="1..8"):
        expr.validate_terms([])
