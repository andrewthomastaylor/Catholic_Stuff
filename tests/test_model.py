import pytest
import numpy as np
from black_scholes_strategy.model import BlackScholesModel

def test_put_call_parity():
    S, K, T, r, sigma = 100, 105, 0.5, 0.05, 0.2
    model = BlackScholesModel(S, K, T, r, sigma)

    call_price = model.calc_price('call')
    put_price = model.calc_price('put')

    # C - P = S - K * exp(-r * T)
    left_side = call_price - put_price
    right_side = S - K * np.exp(-r * T)

    assert pytest.approx(left_side, rel=1e-5) == right_side

def test_intrinsic_value_at_expiry():
    # Call
    S, K, T, r, sigma = 110, 100, 0, 0.05, 0.2
    model = BlackScholesModel(S, K, T, r, sigma)
    assert model.calc_price('call') == 10.0
    assert model.calc_price('put') == 0.0

    # Put
    S, K = 90, 100
    model = BlackScholesModel(S, K, T, r, sigma)
    assert model.calc_price('call') == 0.0
    assert model.calc_price('put') == 10.0

def test_greeks_signs():
    S, K, T, r, sigma = 100, 100, 0.5, 0.05, 0.2
    model = BlackScholesModel(S, K, T, r, sigma)

    # Call Delta should be between 0 and 1
    assert 0 < model.calc_delta('call') < 1
    # Put Delta should be between -1 and 0
    assert -1 < model.calc_delta('put') < 0
    # Gamma should be positive
    assert model.calc_gamma() > 0
    # Vega should be positive
    assert model.calc_vega() > 0
    # Theta is usually negative for long options
    assert model.calc_theta('call') < 0
    assert model.calc_theta('put') < 0 # Can be positive for deep ITM puts, but here it should be negative

def test_model_initialization():
    model = BlackScholesModel(100, 100, 1, 0.05, 0.2)
    assert model.S == 100
    assert model.K == 100
    assert model.T == 1
    assert model.r == 0.05
    assert model.sigma == 0.2

def test_invalid_option_type():
    model = BlackScholesModel(100, 100, 1, 0.05, 0.2)
    with pytest.raises(ValueError):
        model.calc_price('invalid')
    with pytest.raises(ValueError):
        model.calc_delta('invalid')
