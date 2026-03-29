import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import joblib
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, classification_report
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib import cm
import os
from dotenv import load_dotenv
import matplotlib

matplotlib.use('Qt5Agg')


# ----------------------------
# FUNCTION DEFINITIONS
# ----------------------------

def compute_hurst(returns, min_lag=5, max_lag=20):
    """Compute Hurst exponent using variance of log returns."""
    returns = np.array(returns)
    if len(returns) < max_lag + 1:
        return np.nan
    lags = range(min_lag, min(max_lag, len(returns) // 2))
    if len(lags) < 2:
        return np.nan
    variances = []
    for lag in lags:
        aggregated = [np.sum(returns[i:i + lag]) for i in range(0, len(returns) - lag, lag)]
        if len(aggregated) < 2:
            continue
        variances.append(np.var(aggregated, ddof=1))
    if len(variances) < 2:
        return np.nan
    try:
        log_lags = np.log(list(lags)[:len(variances)])
        log_vars = np.log(variances)
        slope = np.polyfit(log_lags, log_vars, 1)[0]
        hurst = slope / 2.0
        return np.clip(hurst, 0.01, 0.99)
    except:
        return np.nan


def load_clean_candles(start='2022-01-01'):
    candles = pd.read_excel('/Users/mikayilmajidov/Projects/Monolith_moextrade/tables/candles.xlsx', index_col=None)
    candles['datetime'] = pd.to_datetime(candles['tradedate'].astype(str) + ' ' + candles['tradetime'].astype(str))
    candles.set_index('datetime', inplace=True)
    candles = candles[candles.index > start]
    candles = candles[candles.index.dayofweek < 5]  # Mon–Fri
    candles = candles.sort_index()

    candles['return'] = candles['pr_close'].pct_change()
    candles['log_return'] = np.log(candles['pr_close'] / candles['pr_close'].shift(1))

    candles['ma_short'] = candles['pr_close'].rolling(5).mean()
    candles['ma_long'] = candles['pr_close'].rolling(20).mean()
    candles['ma_bullish'] = (candles['ma_short'] > candles['ma_long']).astype(float)

    return candles


def compute_daily_features_from_intraday(df_intraday):
    df_intraday = df_intraday.copy()
    required = ['pr_close', 'vol_b', 'vol_s', 'trades_b', 'trades_s']
    for col in required:
        if col not in df_intraday.columns:
            df_intraday[col] = 0

    daily_records = []
    for day, day_data in df_intraday.groupby(df_intraday.index.date):
        if len(day_data) < 10:
            continue

        closes = day_data['pr_close'].values
        time_idx = np.arange(len(closes))
        rho, _ = spearmanr(time_idx, closes)
        spearman_trend = rho

        log_rets = day_data['log_return'].values[1:]
        hurst = compute_hurst(log_rets, min_lag=5, max_lag=20)
        if np.isnan(hurst):
            hurst = 0.5

        returns = day_data['return'].values[1:]
        lag1_autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) >= 3 else 0.0
        volatility = np.std(log_rets, ddof=1) if len(log_rets) > 1 else 0.0

        vol_b = day_data['vol_b'].sum()
        vol_s = day_data['vol_s'].sum()
        trades_b = day_data['trades_b'].sum()
        trades_s = day_data['trades_s'].sum()

        total_vol = vol_b + vol_s
        avg_disb = (vol_b - vol_s) / (total_vol + 1e-9)
        total_trades = trades_b + trades_s
        trade_imb = (trades_b - trades_s) / (total_trades + 1e-9)
        buy_share = vol_b / (total_vol + 1e-9)
        vol_concentration = buy_share

        ma_short = day_data['ma_short'].iloc[-1]
        ma_long = day_data['ma_long'].iloc[-1]
        ma_diff = ma_short - ma_long
        ma_crossover = 1.0 if ma_short > ma_long else 0.0

        dow = pd.Timestamp(day).dayofweek
        dow_sin = np.sin(2 * np.pi * dow / 5.0)
        dow_cos = np.cos(2 * np.pi * dow / 5.0)

        day_high = day_data['pr_high'].max()
        day_low = day_data['pr_low'].min()
        day_open = day_data['pr_open'].iloc[0]
        norm_range = (day_high - day_low) / day_open if day_open != 0 else 0.0

        # ALPHA ENHANCEMENT: Add microstructure features
        # Price momentum (reversal tendency)
        first_half_ret = (closes[len(closes) // 2] - closes[0]) / closes[0] if closes[0] != 0 else 0
        second_half_ret = (closes[-1] - closes[len(closes) // 2]) / closes[len(closes) // 2] if closes[
                                                                                                    len(closes) // 2] != 0 else 0
        intraday_reversal = first_half_ret * second_half_ret  # negative = reversal pattern

        # Volume-weighted trend
        if 'vol_b' in day_data.columns and 'vol_s' in day_data.columns:
            vwap = (day_data['pr_close'] * (day_data['vol_b'] + day_data['vol_s'])).sum() / (
                        day_data['vol_b'] + day_data['vol_s']).sum()
            vwap_distance = (closes[-1] - vwap) / vwap if vwap != 0 else 0
        else:
            vwap_distance = 0

        # Tick intensity (trades per unit time)
        tick_intensity = len(day_data) / (len(day_data) + 1)  # normalized

        daily_records.append({
            'date': pd.Timestamp(day),
            'spearman_trend': spearman_trend,
            'hurst': hurst,
            'lag1_autocorr': lag1_autocorr,
            'volatility': volatility,
            'avg_disb': avg_disb,
            'trade_imb': trade_imb,
            'vol_concentration': vol_concentration,
            'norm_range': norm_range,
            'ma_diff': ma_diff,
            'ma_crossover': ma_crossover,
            'dow_sin': dow_sin,
            'dow_cos': dow_cos,
            'intraday_reversal': intraday_reversal,
            'vwap_distance': vwap_distance,
            'tick_intensity': tick_intensity
        })

    df_daily = pd.DataFrame(daily_records)
    df_daily.set_index('date', inplace=True)
    df_daily.sort_index(inplace=True)
    return df_daily


def assign_daily_regime_expanding(df):
    """
    FIX: Use expanding window to avoid look-ahead bias in quantile calculations
    """
    df = df.copy()
    df['regime'] = 'Undefined'

    for i in range(30, len(df)):  # Need min 30 days for stable quantiles
        # Only use data up to current point
        hist_data = df.iloc[:i + 1]

        current_row = df.iloc[i]

        # Calculate quantiles only on historical data
        norm_range_30 = hist_data['norm_range'].quantile(0.3)
        norm_range_60 = hist_data['norm_range'].quantile(0.6)

        uptrend_condition = (
                (current_row['spearman_trend'] > 0.75) &
                (current_row['hurst'] > 0.5) &
                (current_row['norm_range'] > norm_range_30)
        )
        downtrend_condition = (
                (current_row['spearman_trend'] < -0.75) &
                (current_row['hurst'] > 0.5) &
                (current_row['norm_range'] > norm_range_30)
        )
        mr_condition = (
                (abs(current_row['spearman_trend']) < 0.25) &
                (current_row['hurst'] < 0.4) &
                (current_row['norm_range'] < norm_range_60)
        )

        if uptrend_condition:
            df.iloc[i, df.columns.get_loc('regime')] = 'Uptrend'
        elif downtrend_condition:
            df.iloc[i, df.columns.get_loc('regime')] = 'Downtrend'
        elif mr_condition:
            df.iloc[i, df.columns.get_loc('regime')] = 'MeanReversion'

    return df


def plot_3d_regimes_enhanced(df, candles, max_range=0.03):
    """
    Enhanced 3D plot with multiple views:
    1. Regime coloring
    2. Future return coloring (alpha signal)
    3. Surface interpolation
    4. Density-based coloring
    """
    # Calculate forward returns for alpha analysis
    daily_close = candles['pr_close'].resample('D').last()
    daily_returns = daily_close.pct_change().shift(-1)  # next day return
    df = df.copy()
    df['forward_return'] = df.index.map(daily_returns.to_dict())

    df_plot = df[df['norm_range'] < max_range].copy()
    df_plot = df_plot.dropna(subset=['forward_return'])

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))

    # Plot 1: Colored by regime
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    colors = {'Uptrend': 'green', 'Downtrend': 'red', 'MeanReversion': 'blue', 'Undefined': 'gray'}
    df_plot['color'] = df_plot['regime'].map(colors).fillna('gray')
    ax1.scatter(
        df_plot['spearman_trend'],
        df_plot['hurst'],
        df_plot['norm_range'],
        c=df_plot['color'],
        alpha=0.6,
        s=30
    )
    ax1.set_xlabel('Spearman Trend')
    ax1.set_ylabel('Hurst Exponent')
    ax1.set_zlabel('Normalized Range')
    ax1.set_title("Regime Classification")

    # Plot 2: Colored by forward returns (ALPHA VIEW)
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    # Normalize returns for color mapping
    returns_norm = (df_plot['forward_return'] - df_plot['forward_return'].min()) / \
                   (df_plot['forward_return'].max() - df_plot['forward_return'].min())
    scatter = ax2.scatter(
        df_plot['spearman_trend'],
        df_plot['hurst'],
        df_plot['norm_range'],
        c=df_plot['forward_return'],
        cmap='RdYlGn',  # Red for negative, Green for positive
        alpha=0.7,
        s=30
    )
    ax2.set_xlabel('Spearman Trend')
    ax2.set_ylabel('Hurst Exponent')
    ax2.set_zlabel('Normalized Range')
    ax2.set_title("Colored by Next-Day Return (Alpha View)")
    plt.colorbar(scatter, ax=ax2, label='Forward Return')

    # Plot 3: Surface interpolation showing expected return
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    # Create grid
    x = df_plot['spearman_trend'].values
    y = df_plot['hurst'].values
    z = df_plot['forward_return'].values

    xi = np.linspace(x.min(), x.max(), 50)
    yi = np.linspace(y.min(), y.max(), 50)
    XI, YI = np.meshgrid(xi, yi)

    # Interpolate
    ZI = griddata((x, y), z, (XI, YI), method='linear')

    surf = ax3.plot_surface(XI, YI, ZI, cmap='RdYlGn', alpha=0.6, edgecolor='none')
    ax3.scatter(x, y, z, c=z, cmap='RdYlGn', s=10, alpha=0.3)
    ax3.set_xlabel('Spearman Trend')
    ax3.set_ylabel('Hurst Exponent')
    ax3.set_zlabel('Expected Return')
    ax3.set_title("Expected Return Surface")
    plt.colorbar(surf, ax=ax3, label='Expected Return')

    # Plot 4: Contour view (top-down)
    ax4 = fig.add_subplot(2, 3, 4)
    contour = ax4.tricontourf(
        df_plot['spearman_trend'],
        df_plot['hurst'],
        df_plot['forward_return'],
        levels=20,
        cmap='RdYlGn'
    )
    ax4.scatter(
        df_plot['spearman_trend'],
        df_plot['hurst'],
        c='black',
        s=5,
        alpha=0.3
    )
    ax4.set_xlabel('Spearman Trend')
    ax4.set_ylabel('Hurst Exponent')
    ax4.set_title("Return Heatmap (Top View)")
    plt.colorbar(contour, ax=ax4, label='Forward Return')

    # Plot 5: Hexbin density
    ax5 = fig.add_subplot(2, 3, 5)
    hexbin = ax5.hexbin(
        df_plot['spearman_trend'],
        df_plot['hurst'],
        C=df_plot['forward_return'],
        gridsize=20,
        cmap='RdYlGn',
        reduce_C_function=np.mean
    )
    ax5.set_xlabel('Spearman Trend')
    ax5.set_ylabel('Hurst Exponent')
    ax5.set_title("Hexbin Density (Avg Return)")
    plt.colorbar(hexbin, ax=ax5, label='Avg Forward Return')

    # Plot 6: Distribution of returns by regime
    ax6 = fig.add_subplot(2, 3, 6)
    for regime in ['Uptrend', 'Downtrend', 'MeanReversion', 'Undefined']:
        regime_data = df_plot[df_plot['regime'] == regime]['forward_return'].dropna()
        if len(regime_data) > 0:
            ax6.hist(regime_data, bins=30, alpha=0.5, label=f'{regime} (μ={regime_data.mean():.4f})',
                     color=colors.get(regime, 'gray'))
    ax6.set_xlabel('Forward Return')
    ax6.set_ylabel('Frequency')
    ax6.set_title("Return Distribution by Regime")
    ax6.legend()
    ax6.axvline(0, color='black', linestyle='--', linewidth=1)

    plt.tight_layout()
    plt.show()

    # Print regime statistics
    print("\n📊 REGIME STATISTICS (Forward Returns):")
    for regime in ['Uptrend', 'Downtrend', 'MeanReversion', 'Undefined']:
        regime_data = df_plot[df_plot['regime'] == regime]
        if len(regime_data) > 0:
            mean_ret = regime_data['forward_return'].mean()
            std_ret = regime_data['forward_return'].std()
            sharpe = mean_ret / std_ret if std_ret > 0 else 0
            win_rate = (regime_data['forward_return'] > 0).mean()
            print(
                f"{regime:15} | Mean: {mean_ret:7.4f} | Std: {std_ret:6.4f} | Sharpe: {sharpe:6.3f} | WinRate: {win_rate:.2%} | N: {len(regime_data)}")


def train_hmm_properly(X_scaled, n_states=4):
    """
    FIX: Train HMM in unsupervised manner, let it discover states
    Then we can analyze what those states represent
    """
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=42,
        verbose=False
    )
    model.fit(X_scaled)

    # Decode the hidden states
    hidden_states = model.predict(X_scaled)

    return model, hidden_states


def compute_one_step_predictions_fixed(X_scaled, model):
    """
    FIX: Proper one-step-ahead prediction using HMM
    For each time t, use observations up to t-1 to predict state at t
    """
    n_samples = len(X_scaled)
    y_pred = np.zeros(n_samples - 1, dtype=int)
    pred_probs = np.zeros((n_samples - 1, model.n_components))

    for t in range(1, n_samples):
        # Use all data up to t-1
        X_past = X_scaled[:t]

        # Decode to get state at t-1
        if len(X_past) > 0:
            _, states = model.decode(X_past, algorithm='viterbi')
            last_state = states[-1]

            # Use transition matrix to predict next state
            next_state_probs = model.transmat_[last_state]
            y_pred[t - 1] = np.argmax(next_state_probs)
            pred_probs[t - 1] = next_state_probs

    return y_pred, pred_probs


def backtest_ma_crossover_fixed(candles_5min, df_result, state_labels=None):
    """
    FIX: Use proper entry prices (open of next bar or limit orders)
    """
    if state_labels is None:
        state_labels = {'Uptrend', 'Downtrend', 'MeanReversion', 'Undefined'}

    df = candles_5min.copy()
    df = df[df.index.dayofweek < 5]

    # Compute MAs
    df['ma_fast'] = df['pr_close'].rolling(6).mean()
    df['ma_slow'] = df['pr_close'].rolling(18).mean()

    # Generate crossover signals
    df['fast_above_slow'] = df['ma_fast'] > df['ma_slow']
    df['crossover_up'] = (df['fast_above_slow']) & (~df['fast_above_slow'].shift(1).fillna(False))
    df['crossover_down'] = (~df['fast_above_slow']) & (df['fast_above_slow'].shift(1).fillna(False))

    # Map daily regime
    df['date'] = df.index.date
    daily_regime_map = df_result['pred_regime'].to_dict()
    df['regime'] = df['date'].map(daily_regime_map)

    df['position'] = 0
    df['pnl'] = 0.0

    current_pos = 0
    entry_price = 0.0
    trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        idx = df.index[i]

        is_last_bar = (i == len(df) - 1) or (df.index[i].date() != df.index[i + 1].date())

        # Close at end of day
        if current_pos != 0 and is_last_bar:
            exit_price = row['pr_close']
            trade_pnl = (exit_price - entry_price) * current_pos
            df.loc[idx, 'pnl'] = trade_pnl
            trades.append({
                'entry_time': entry_idx,
                'exit_time': idx,
                'side': 'long' if current_pos > 0 else 'short',
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': trade_pnl
            })
            current_pos = 0
            entry_price = 0.0

        # Open new position
        if current_pos == 0:
            regime = row['regime']
            if pd.isna(regime):
                continue

            # FIX: Use OPEN of next bar as entry price (or current close + slippage)
            # This is more realistic
            if i < len(df) - 1:
                next_bar_open = df.iloc[i + 1]['pr_open']
            else:
                next_bar_open = row['pr_close']

            if regime == 'Uptrend' and prev_row['crossover_up']:
                current_pos = 1
                entry_price = next_bar_open  # FIX: realistic entry
                entry_idx = idx

            elif regime == 'Downtrend' and prev_row['crossover_down']:
                current_pos = -1
                entry_price = next_bar_open  # FIX: realistic entry
                entry_idx = idx

    df['cum_pnl'] = df['pnl'].cumsum()

    trades_df = pd.DataFrame(trades)
    total_pnl = df['pnl'].sum()

    return {
        'equity_curve': df['cum_pnl'],
        'trades': trades_df,
        'total_pnl': total_pnl,
        'df_with_signals': df
    }


# ----------------------------
# ALPHA ENHANCEMENTS
# ----------------------------

def add_alpha_features(df_daily):
    """
    Add features that could generate alpha:
    1. Regime transition probability
    2. Volatility regime
    3. Order flow imbalance momentum
    4. Mean reversion indicators
    """
    df = df_daily.copy()

    # Volatility regime (expanding percentile)
    df['vol_percentile'] = df['volatility'].expanding(min_periods=30).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if len(x) > 0 else 0.5
    )

    # Order flow momentum (3-day moving average)
    df['order_flow_ma3'] = df['avg_disb'].rolling(3).mean()
    df['order_flow_momentum'] = df['avg_disb'] - df['order_flow_ma3']

    # Hurst momentum (is it trending toward mean reversion?)
    df['hurst_change'] = df['hurst'].diff(1)

    # Autocorrelation regime
    df['autocorr_ma5'] = df['lag1_autocorr'].rolling(5).mean()

    # Combined momentum/mean-reversion score
    df['momentum_score'] = (
            df['spearman_trend'].rolling(3).mean() *
            df['hurst'] *
            (1 + df['order_flow_momentum'].fillna(0))
    )

    return df


# ----------------------------
# MAIN EXECUTION
# ----------------------------
if __name__ == "__main__":
    # Load and prepare data
    candles = load_clean_candles(start='2022-01-21')
    df_daily = compute_daily_features_from_intraday(candles)

    # Add alpha features
    df_daily = add_alpha_features(df_daily)

    # FIX: Use expanding window for regime assignment
    df_daily = assign_daily_regime_expanding(df_daily)

    # Enhanced 3D visualization
    print("Generating enhanced 3D visualizations...")
    plot_3d_regimes_enhanced(df_daily, candles, max_range=0.03)

    print("\nRegime counts:")
    print(df_daily['regime'].value_counts())

    # Prepare features for HMM
    feature_cols = [
        'spearman_trend', 'hurst', 'norm_range',
        'lag1_autocorr', 'volatility', 'avg_disb',
        'trade_imb', 'ma_diff', 'dow_sin', 'dow_cos',
        # Alpha features
        'intraday_reversal', 'vwap_distance',
        'vol_percentile', 'order_flow_momentum', 'hurst_change'
    ]

    df_ml = df_daily.dropna(subset=feature_cols).copy()

    # Scale features
    X = df_ml[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # FIX: Train HMM properly (unsupervised)
    n_states = 4
    model, hidden_states = train_hmm_properly(X_scaled, n_states=n_states)

    # Analyze what the HMM discovered
    df_ml['hmm_state'] = hidden_states

    print("\nHMM STATE ANALYSIS:")
    for state in range(n_states):
        state_data = df_ml[df_ml['hmm_state'] == state]
        print(f"\nState {state} (N={len(state_data)}):")
        print(f"  Spearman: {state_data['spearman_trend'].mean():6.3f}")
        print(f"  Hurst:    {state_data['hurst'].mean():6.3f}")
        print(f"  Vol:      {state_data['volatility'].mean():6.4f}")
        print(f"  OrderFlow:{state_data['avg_disb'].mean():6.3f}")

    # Compute one-step-ahead predictions
    y_pred, pred_probs = compute_one_step_predictions_fixed(X_scaled, model)
    y_true = hidden_states[1:]

    # Create results dataframe
    dates = df_ml.index[1:]
    df_result = pd.DataFrame(index=dates)
    df_result['true_state'] = y_true
    df_result['pred_state'] = y_pred

    for i in range(n_states):
        df_result[f'prob_state_{i}'] = pred_probs[:, i]

    # Map to regime labels based on characteristics
    # (You would analyze HMM states and assign meaningful labels)
    state_to_regime = {0: 'Uptrend', 1: 'Downtrend', 2: 'MeanReversion', 3: 'Undefined'}
    df_result['pred_regime'] = df_result['pred_state'].map(state_to_regime)
    df_result['true_regime'] = df_result['true_state'].map(state_to_regime)

    print(f"\n🎯 One-day-ahead accuracy: {accuracy_score(y_true, y_pred):.2%}")

    # Save results
    df_result.to_pickle('daily_hmm_predictions_fixed.pkl')
    joblib.dump(model, 'hmm_daily_model_fixed.pkl')
    joblib.dump(scaler, 'hmm_daily_scaler_fixed.pkl')

    print("\n✅ Improved model saved!")