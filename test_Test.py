import pytest
import numpy as np
from Test import BlackScholesOptionPricer, calculate_days_to_expiry, greet
from datetime import datetime, timedelta



class TestBlackScholesOptionPricer:




    @pytest.fixture
    def pricer(self):
        return BlackScholesOptionPricer(
            spot_price=100.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            dividend_yield=0.01
        )



<span class="hljs-keyword">def</span> <span class="hljs-title function_">test_call_price</span>(<span class="hljs-params">self, pricer</span>):
    call_price = pricer.call_price()
    <span class="hljs-keyword">assert</span> <span class="hljs-built_in">isinstance</span>(call_price, <span class="hljs-built_in">float</span>)
    <span class="hljs-keyword">assert</span> call_price &gt; <span class="hljs-number">0</span>
    <span class="hljs-comment"># Test with known value (approximate)</span>
    <span class="hljs-keyword">assert</span> <span class="hljs-built_in">abs</span>(call_price - <span class="hljs-number">12.2</span>) &lt; <span class="hljs-number">1.0</span>

<span class="hljs-keyword">def</span> <span class="hljs-title function_">test_put_price</span>(<span class="hljs-params">self, pricer</span>):
    put_price = pricer.put_price()
    <span class="hljs-keyword">assert</span> <span class="hljs-built_in">isinstance</span>(put_price, <span class="hljs-built_in">float</span>)
    <span class="hljs-keyword">assert</span> put_price &gt; <span class="hljs-number">0</span>
    <span class="hljs-comment"># Test with known value (approximate)</span>
    <span class="hljs-keyword">assert</span> <span class="hljs-built_in">abs</span>(put_price - <span class="hljs-number">7.7</span>) &lt; <span class="hljs-number">1.0</span>

<span class="hljs-keyword">def</span> <span class="hljs-title function_">test_call_delta</span>(<span class="hljs-params">self, pricer</span>):
    delta = pricer.call_delta()
    <span class="hljs-keyword">assert</span> <span class="hljs-built_in">isinstance</span>(delta, <span class="hljs-built_in">float</span>)
    <span class="hljs-keyword">assert</span> <span class="hljs-number">0</span> &lt;= delta &lt;= <span class="hljs-number">1</span>

<span class="hljs-keyword">def</span> <span class="hljs-title function_">test_put_delta</span>(<span class="hljs-params">self, pricer</span>):
    delta = pricer.put_delta()
    <span class="hljs-keyword">assert</span> <span class="hljs-built_in">isinstance</span>(delta, <span class="hljs-built_in">float</span>)
    <span class="hljs-keyword">assert</span> -<span class="hljs-number">1</span> &lt;= delta &lt;= <span class="hljs-number">0</span>

<span class="hljs-keyword">def</span> <span class="hljs-title function_">test_gamma</span>(<span class="hljs-params">self, pricer</span>):
    gamma = pricer.gamma()
    <span class="hljs-keyword">assert</span> <span class="hljs-built_in">isinstance</span>(gamma, <span class="hljs-built_in">float</span>)
    <span class="hljs-keyword">assert</span> gamma &gt; <span class="hljs-number">0</span>

<span class="hljs-keyword">def</span> <span class="hljs-title function_">test_vega</span>(<span class="hljs-params">self, pricer</span>):
    vega = pricer.vega()
    <span class="hljs-keyword">assert</span> <span class="hljs-built_in">isinstance</span>(vega, <span class="hljs-built_in">float</span>)
    <span class="hljs-keyword">assert</span> vega &gt; <span class="hljs-number">0</span>

<span class="hljs-keyword">def</span> <span class="hljs-title function_">test_call_theta</span>(<span class="hljs-params">self, pricer</span>):
    theta = pricer.call_theta()
    <span class="hljs-keyword">assert</span> <span class="hljs-built_in">isinstance</span>(theta, <span class="hljs-built_in">float</span>)
    <span class="hljs-comment"># Theta is typically negative for calls</span>
    <span class="hljs-keyword">assert</span> theta &lt; <span class="hljs-number">0</span>

<span class="hljs-keyword">def</span> <span class="hljs-title function_">test_put_theta</span>(<span class="hljs-params">self, pricer</span>):
    theta = pricer.put_theta()
    <span class="hljs-keyword">assert</span> <span class="hljs-built_in">isinstance</span>(theta, <span class="hljs-built_in">float</span>)

<span class="hljs-keyword">def</span> <span class="hljs-title function_">test_call_rho</span>(<span class="hljs-params">self, pricer</span>):
    rho = pricer.call_rho()
    <span class="hljs-keyword">assert</span> <span class="hljs-built_in">isinstance</span>(rho, <span class="hljs-built_in">float</span>)
    <span class="hljs-keyword">assert</span> rho &gt; <span class="hljs-number">0</span>

<span class="hljs-keyword">def</span> <span class="hljs-title function_">test_put_rho</span>(<span class="hljs-params">self, pricer</span>):
    rho = pricer.put_rho()
    <span class="hljs-keyword">assert</span> <span class="hljs-built_in">isinstance</span>(rho, <span class="hljs-built_in">float</span>)
    <span class="hljs-keyword">assert</span> rho &lt; <span class="hljs-number">0</span>

<span class="hljs-keyword">def</span> <span class="hljs-title function_">test_implied_volatility</span>(<span class="hljs-params">self, pricer</span>):
    <span class="hljs-comment"># Test with a known option price</span>
    call_price = pricer.call_price()
    implied_vol = pricer.implied_volatility(call_price, <span class="hljs-string">&#x27;call&#x27;</span>)
    <span class="hljs-keyword">assert</span> <span class="hljs-built_in">abs</span>(implied_vol - <span class="hljs-number">0.2</span>) &lt; <span class="hljs-number">0.01</span>




def test_calculate_days_to_expiry():
    # Test with a future date
    today = datetime.now().date()
    future_date = (today + timedelta(days=30)).strftime('%Y-%m-%d')
    days_to_expiry = calculate_days_to_expiry(future_date)
    assert isinstance(days_to_expiry, float)
    assert abs(days_to_expiry - 30/365) < 0.01


