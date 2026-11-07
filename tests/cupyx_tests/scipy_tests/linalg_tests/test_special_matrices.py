from __future__ import annotations

import unittest
import warnings

import pytest

from cupy import testing
import cupyx.scipy.linalg  # NOQA

try:
    import scipy.linalg  # NOQA
except ImportError:
    pass


class TestSpecialMatricesBase(unittest.TestCase):
    def _get_arg(self, xp, arg):
        if isinstance(arg, tuple):
            # Allocate array with the given shape
            return testing.shaped_random(arg, xp)

        # Otherwise just pass the arg back
        return arg


@testing.parameterize(*(
    testing.product({
        # 1 argument: 1D array
        'function': ['circulant', 'toeplitz', 'hankel', 'companion'],
        'args': [((1,),), ((2,),), ((4,),), ((10,),), ((25,),)],
    }) + testing.product({
        # 2 arguments: both 1D arrays
        # For leslie, second array should be 1 less long than first
        'function': ['toeplitz', 'hankel', 'leslie'],
        'args': [((1,), (1,)), ((2,), (1,)), ((4,), (5,)),
                 ((10,), (9,)), ((25,), (24,))],
    }) + testing.product({
        # 1 argument: int
        'function': ['hadamard', 'helmert', 'hilbert', 'dft'],
        'args': [(1,), (2,), (4,), (10,), (25,)],
    }) + testing.product({
        # 2 arguments: int, dtype
        'function': ['hadamard'],
        'args': [(4, 'int32'), (8, 'float64'), (6, 'float64')],
    }) + testing.product({
        # 2 arguments: int, bool
        'function': ['helmert'],
        'args': [(4, True), (5, False)],
    }) + testing.product({
        # 2 arguments: int, str
        'function': ['dft'],
        'args': [(4, 'sqrtn'), (5, 'n')],
    }) + testing.product({
        # 2 arguments: both 2D arrays
        'function': ['kron'],
        'args': [((5, 5), (4, 5),), ((5, 4), (4, 4),), ((1, 2), (4, 5))],
    }) + testing.product({
        # 0 or more arguments: all 1 or 2D arrays
        'function': ['block_diag'],
        'args': [(), ((5,),), ((4, 5),), ((5,), (4, 4),), ((1, 2), (4, 5)),
                 ((1,), (4, 5), (3, 6), (7, 2), (8, 9))],
    })
))
@testing.with_requires('scipy')
class TestSpecialMatrices(TestSpecialMatricesBase):
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp',
                                 accept_error=ValueError)
    def test_special_matrix(self, xp, scp):
        if self.function == "kron" and not testing.installed("scipy<1.17"):
            self.skipTest("scipy.linalg.kron was removed in scipy 1.17")
        function = getattr(scp.linalg, self.function)

        if self.function == "kron":
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=DeprecationWarning)
                return function(*[self._get_arg(xp, arg) for arg in self.args])
        else:
            return function(*[self._get_arg(xp, arg) for arg in self.args])


@testing.parameterize(*(
    testing.product({
        # 1 argument: 1D array
        'function': ['fiedler', 'fiedler_companion'],
        'args': [((0,),), ((1,),), ((2,),), ((4,),), ((10,),), ((25,),)],
    })
))
@testing.with_requires('scipy')
class TestSpecialMatrices_1_3_0(TestSpecialMatricesBase):
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp',
                                 accept_error=ValueError)
    def test_special_matrix(self, xp, scp):
        # Both functions build an `(n, n)` matrix, so the degenerate inputs
        # below must give `(0, 0)`. CuPy returns that shape, as does SciPy
        # 1.18 and later, but older SciPy returned a 1-D empty array.
        degenerate = ((self.function == 'fiedler' and self.args == ((0,),))
                      or (self.function == 'fiedler_companion'
                          and self.args == ((1,),)))
        if degenerate and testing.installed('scipy<1.18'):
            pytest.xfail('SciPy <1.18 returns a 1-D empty array here')
        function = getattr(scp.linalg, self.function)
        return function(*[self._get_arg(xp, arg) for arg in self.args])

    def _get_arg(self, xp, arg):
        if isinstance(arg, tuple):
            # Allocate array with the given shape
            return testing.shaped_random(arg, xp)

        # Otherwise just pass the arg back
        return arg


class TestFiedlerDegenerate:
    # `TestSpecialMatrices_1_3_0` cannot compare these against SciPy <1.18,
    # so pin the shape down here instead.
    @pytest.mark.parametrize(('function', 'n'),
                             [('fiedler', 0), ('fiedler_companion', 1)])
    def test_empty_matrix(self, function, n):
        a = testing.shaped_random((n,))
        assert getattr(cupyx.scipy.linalg, function)(a).shape == (0, 0)

    def test_fiedler_companion_empty_input(self):
        # SciPy returns a 1-D empty array for a size-0 coefficient array.
        assert cupyx.scipy.linalg.fiedler_companion(
            testing.shaped_random((0,))).shape == (0,)


@testing.parameterize(*(
    testing.product({
        # 2-3 arguments: 1D array, 1 int, 1 optional str
        'function': ['convolution_matrix'],
        'args': [((1,), 5), ((2,), 3), ((4,), 10), ((10,), 15), ((25,), 25),
                 ((4,), 6, 'full'), ((10,), 8, 'same'), ((25,), 25, 'valid')],
    })
))
@testing.with_requires('scipy')
class TestSpecialMatrices_1_5_0(TestSpecialMatricesBase):
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp',
                                 accept_error=ValueError)
    def test_special_matrix(self, xp, scp):
        function = getattr(scp.linalg, self.function)
        return function(*[self._get_arg(xp, arg) for arg in self.args])


@testing.with_requires('scipy>=1.15')
class TestSpecialMatricesBatch:
    # SciPy treats N-D input as a batch: each slice along the last axis gives
    # one matrix.

    @pytest.mark.parametrize('batch', [(3,), (2, 3), (2, 1, 3)])
    @pytest.mark.parametrize(('function', 'shapes'), [
        ('circulant', [(5,)]),
        ('companion', [(5,)]),
        ('fiedler', [(5,)]),
        ('fiedler_companion', [(2,)]),
        ('fiedler_companion', [(6,)]),
        ('fiedler_companion', [(7,)]),
        ('leslie', [(5,), (4,)]),
    ])
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_batch(self, xp, scp, function, shapes, batch):
        args = [testing.shaped_random(batch + shape, xp, scale=5, seed=i)
                for i, shape in enumerate(shapes)]
        return getattr(scp.linalg, function)(*args)

    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_leslie_broadcast(self, xp, scp):
        f = testing.shaped_random((2, 3, 5), xp, seed=0)
        s = testing.shaped_random((4,), xp, seed=1)
        return scp.linalg.leslie(f, s)

    @pytest.mark.parametrize('mode', ['full', 'same', 'valid'])
    @pytest.mark.parametrize('n', [2, 7])
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_convolution_matrix(self, xp, scp, n, mode):
        a = testing.shaped_random((2, 3, 5), xp)
        return scp.linalg.convolution_matrix(a, n, mode)

    @testing.for_complex_dtypes()
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_fiedler_complex(self, xp, scp, dtype):
        # The absolute value of a complex difference is real.
        a = testing.shaped_random((2, 4), xp, dtype)
        return scp.linalg.fiedler(a)


@testing.with_requires('scipy>=1.17')
class TestToeplitzBatch:
    # SciPy 1.17 treats N-D input to `toeplitz` as a batch instead of
    # flattening it.

    @pytest.mark.parametrize(('c_shape', 'r_shape'), [
        ((2, 3, 4), None),
        ((2, 3, 4), (2, 3, 6)),
        ((2, 3, 4), (6,)),
        ((4,), (2, 3, 6)),
        ((3, 1, 4), (1, 2, 5)),
    ])
    @testing.numpy_cupy_allclose(atol=1e-5, rtol=1e-5, scipy_name='scp')
    def test_toeplitz(self, xp, scp, c_shape, r_shape):
        c = testing.shaped_random(c_shape, xp, seed=0)
        if r_shape is None:
            return scp.linalg.toeplitz(c)
        r = testing.shaped_random(r_shape, xp, seed=1)
        return scp.linalg.toeplitz(c, r)
