"""
Prop Firm Evaluation Trading Bot - QuantConnect Compatible
Designed for Topstep/Apex Trader Funding evaluations

DISCLAIMER: Only uses publicly available information. No insider trading.
"""

from AlgorithmImports import *
import numpy as np
from datetime import timedelta, time
from collections import deque


class PropFirmEvaluationBot(QCAlgorithm):
    """
    Multi-strategy futures trading bot optimized for prop firm evaluations.
    Focuses on risk management and consistency over aggressive returns.
    """

    def Initialize(self):
        # === CONFIGURATION ===
        self.SetStartDate(2023, 1, 1)
        self.SetEndDate(2024, 10, 1)
        self.SetCash(50000)  # Typical evaluation account size
        self.initial_cash = 50000  # Store for P&L calculations

        # Primary futures contract (ES - E-mini S&P 500)
        self.futures_symbol = self.AddFuture(
            Futures.Indices.SP500EMini,
            Resolution.Minute,
            dataNormalizationMode=DataNormalizationMode.BackwardsPanamaCanal
        )
        self.futures_symbol.SetFilter(0, 182)  # Front month

        # Store active contract
        self.contract = None

        # === PROP FIRM RULES ===
        self.max_daily_loss = 1000  # Typical $1000 trailing loss limit
        self.max_total_loss = 2000  # Typical $2000 max loss limit
        self.profit_target = 3000  # Typical $3000 profit target
        self.max_contracts = 2  # Position sizing limit
        self.daily_start_equity = self.Portfolio.TotalPortfolioValue

        # === INDICATORS ===
        self.ema_fast = None
        self.ema_slow = None
        self.atr = None
        self.rsi = None
        self.bb = None
        self.adx = None
        self.volume_ema = None

        # === STATE MANAGEMENT ===
        self.recent_trades = deque(maxlen=20)
        self.win_streak = 0
        self.loss_streak = 0
        self.daily_pnl = 0
        self.session_high_equity = self.Portfolio.TotalPortfolioValue

        # === NEWS & EVENTS ===
        # QuantConnect's economic calendar for FOMC, NFP, etc.
        self.economic_events = []
        self.news_sentiment = 0  # -1 to 1 scale

        # High-impact event times (adjust for EST)
        self.fomc_days = []  # Will be populated
        self.blackout_until = None

        # Market regime detection
        self.regime = "neutral"  # trending_up, trending_down, ranging, volatile
        self.regime_history = deque(maxlen=100)

        # === RISK PARAMETERS ===
        self.position_score = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0

        # Schedule functions
        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.At(9, 30),  # Market open
            self.OnMarketOpen
        )

        self.Schedule.On(
            self.DateRules.EveryDay(),
            self.TimeRules.BeforeMarketClose(self.futures_symbol.Symbol, 30),
            self.BeforeMarketClose
        )

        # Warm up period
        self.SetWarmUp(timedelta(days=30))

    def OnData(self, data):
        """Main trading logic"""

        if self.IsWarmingUp:
            return

        # Update active contract
        self._update_contract(data)

        if self.contract is None or not data.ContainsKey(self.contract):
            return

        # Initialize indicators if needed
        if self.ema_fast is None:
            self._initialize_indicators()
            return

        if not self.ema_fast.IsReady:
            return

        # Update daily PnL tracking
        self._update_risk_metrics()

        # Check prop firm rule violations
        if self._check_rule_violations():
            self.Liquidate()
            return

        # Check if in blackout period (major news events)
        if self._is_blackout_period():
            if self.Portfolio.Invested:
                self.Liquidate()
            return

        # Update market regime
        self._update_market_regime()

        # Get current price
        price = self.Securities[self.contract].Price

        # Execute trading strategy
        if not self.Portfolio.Invested:
            self._check_entry_signals(price)
        else:
            self._manage_position(price)

    def _update_contract(self, data):
        """Update to the front month contract"""
        for chain in data.FutureChains:
            contracts = [c for c in chain.Value if c.Expiry > self.Time]
            if contracts:
                # Select front month
                self.contract = sorted(contracts, key=lambda x: x.Expiry)[0].Symbol

    def _initialize_indicators(self):
        """Initialize technical indicators"""
        if self.contract is None:
            return

        # Trend following
        self.ema_fast = self.EMA(self.contract, 9)
        self.ema_slow = self.EMA(self.contract, 21)

        # Volatility & momentum
        self.atr = self.ATR(self.contract, 14)
        self.rsi = self.RSI(self.contract, 14)
        self.adx = self.ADX(self.contract, 14)

        # Mean reversion
        self.bb = self.BB(self.contract, 20, 2)

        # Volume indicator - using simple consolidator approach
        self.volume_ema = None  # Will track manually in OnData
        self.volume_history = deque(maxlen=20)

    def _update_risk_metrics(self):
        """Track daily P&L and equity high water mark"""
        current_equity = self.Portfolio.TotalPortfolioValue

        # Update session high
        if current_equity > self.session_high_equity:
            self.session_high_equity = current_equity

        # Calculate daily P&L
        self.daily_pnl = current_equity - self.daily_start_equity

    def _check_rule_violations(self):
        """Check if any prop firm rules are violated"""
        current_equity = self.Portfolio.TotalPortfolioValue
        total_pnl = current_equity - self.initial_cash

        # Trailing drawdown from session high
        trailing_dd = self.session_high_equity - current_equity

        # Check violations
        if trailing_dd >= self.max_daily_loss:
            self.Debug(f"VIOLATION: Daily loss limit hit - ${trailing_dd:.2f}")
            return True

        if total_pnl <= -self.max_total_loss:
            self.Debug(f"VIOLATION: Max total loss hit - ${total_pnl:.2f}")
            return True

        # Check if profit target reached (some evaluations)
        if total_pnl >= self.profit_target:
            self.Debug(f"SUCCESS: Profit target reached - ${total_pnl:.2f}")
            self.Liquidate()
            return True

        return False

    def _is_blackout_period(self):
        """Check if during high-impact news event"""
        if self.blackout_until and self.Time < self.blackout_until:
            return True

        # Check for FOMC days (2:00 PM ET announcements)
        current_date = self.Time.date()
        if current_date in self.fomc_days:
            # Blackout 30 min before and after
            announcement_time = time(14, 0)  # 2 PM ET
            if time(13, 30) <= self.Time.time() <= time(14, 30):
                return True

        # NFP (First Friday, 8:30 AM ET) - avoid 8:15-9:00 AM
        if self.Time.weekday() == 4:  # Friday
            if self.Time.day <= 7:  # First week
                if time(8, 15) <= self.Time.time() <= time(9, 0):
                    return True

        return False

    def _update_market_regime(self):
        """Detect market regime for strategy adaptation"""
        if not self.adx.IsReady:
            return

        adx_value = self.adx.Current.Value
        rsi_value = self.rsi.Current.Value
        atr_value = self.atr.Current.Value

        # Trending market (ADX > 25)
        if adx_value > 25:
            if self.ema_fast.Current.Value > self.ema_slow.Current.Value:
                self.regime = "trending_up"
            else:
                self.regime = "trending_down"
        # Ranging market (ADX < 20)
        elif adx_value < 20:
            self.regime = "ranging"
        # High volatility
        elif atr_value > self.atr.IsReady and atr_value > 1.5 * np.mean([x.Value for x in list(self.atr)[-10:]]):
            self.regime = "volatile"
        else:
            self.regime = "neutral"

        self.regime_history.append(self.regime)

    def _check_entry_signals(self, price):
        """Multi-strategy entry signal generation"""

        # Position sizing based on account size and risk
        contracts = self._calculate_position_size()
        if contracts == 0:
            return

        # === STRATEGY 1: Trend Following with Momentum ===
        trend_score = self._trend_following_signal()

        # === STRATEGY 2: Mean Reversion ===
        reversion_score = self._mean_reversion_signal()

        # === STRATEGY 3: Breakout ===
        breakout_score = self._breakout_signal()

        # Combine signals with regime weighting
        if self.regime in ["trending_up", "trending_down"]:
            self.position_score = trend_score * 0.6 + breakout_score * 0.4
        elif self.regime == "ranging":
            self.position_score = reversion_score * 0.7 + trend_score * 0.3
        else:
            self.position_score = (trend_score + reversion_score + breakout_score) / 3

        # Entry thresholds
        long_threshold = 0.6
        short_threshold = -0.6

        # Execute trades
        if self.position_score > long_threshold and self.loss_streak < 3:
            self._enter_long(contracts, price)
        elif self.position_score < short_threshold and self.loss_streak < 3:
            self._enter_short(contracts, price)

    def _trend_following_signal(self):
        """Trend following strategy score"""
        score = 0

        # EMA crossover
        if self.ema_fast.Current.Value > self.ema_slow.Current.Value:
            score += 0.3
        else:
            score -= 0.3

        # RSI momentum
        rsi_val = self.rsi.Current.Value
        if rsi_val > 50 and rsi_val < 70:
            score += 0.2
        elif rsi_val < 50 and rsi_val > 30:
            score -= 0.2

        # ADX strength
        if self.adx.Current.Value > 25:
            score *= 1.5  # Amplify in trending markets

        return np.clip(score, -1, 1)

    def _mean_reversion_signal(self):
        """Mean reversion strategy score"""
        score = 0
        price = self.Securities[self.contract].Price

        # Bollinger Bands
        if price < self.bb.LowerBand.Current.Value:
            score += 0.4  # Oversold
        elif price > self.bb.UpperBand.Current.Value:
            score -= 0.4  # Overbought

        # RSI extremes
        rsi_val = self.rsi.Current.Value
        if rsi_val < 30:
            score += 0.3
        elif rsi_val > 70:
            score -= 0.3

        return np.clip(score, -1, 1)

    def _breakout_signal(self):
        """Breakout strategy score"""
        score = 0
        price = self.Securities[self.contract].Price
        current_vol = self.Securities[self.contract].Volume

        # Track volume history
        self.volume_history.append(current_vol)

        # Volume confirmation
        if len(self.volume_history) >= 20:
            avg_vol = np.mean(self.volume_history)

            if current_vol > 1.5 * avg_vol:
                # High volume breakout
                if price > self.bb.UpperBand.Current.Value:
                    score += 0.5
                elif price < self.bb.LowerBand.Current.Value:
                    score -= 0.5

        return np.clip(score, -1, 1)

    def _calculate_position_size(self):
        """Risk-based position sizing"""
        # Conservative sizing for prop firms
        account_value = self.Portfolio.TotalPortfolioValue

        # Risk 0.5% per trade
        risk_per_trade = account_value * 0.005

        # Calculate based on ATR stop loss
        if self.atr.IsReady:
            atr_value = self.atr.Current.Value
            stop_distance = 2 * atr_value  # 2 ATR stop

            # ES contract multiplier is $50
            contracts = int(risk_per_trade / (stop_distance * 50))

            # Cap at max allowed
            return min(contracts, self.max_contracts)

        return 1  # Default to 1 contract

    def _enter_long(self, contracts, price):
        """Enter long position with risk management"""
        self.MarketOrder(self.contract, contracts)

        self.entry_price = price
        atr_val = self.atr.Current.Value

        # Set stops
        self.stop_loss = price - (2 * atr_val)
        self.take_profit = price + (3 * atr_val)  # 1.5:1 reward/risk

        self.Debug(f"LONG: {contracts} @ {price:.2f} | SL: {self.stop_loss:.2f} | TP: {self.take_profit:.2f}")

    def _enter_short(self, contracts, price):
        """Enter short position with risk management"""
        self.MarketOrder(self.contract, -contracts)

        self.entry_price = price
        atr_val = self.atr.Current.Value

        # Set stops
        self.stop_loss = price + (2 * atr_val)
        self.take_profit = price - (3 * atr_val)

        self.Debug(f"SHORT: {contracts} @ {price:.2f} | SL: {self.stop_loss:.2f} | TP: {self.take_profit:.2f}")

    def _manage_position(self, price):
        """Active position management"""
        position = self.Portfolio[self.contract]

        if position.IsLong:
            # Check stop loss
            if price <= self.stop_loss:
                self.Liquidate(self.contract)
                self._record_trade(False)
                self.Debug(f"STOP LOSS HIT: {price:.2f}")
                return

            # Check take profit
            if price >= self.take_profit:
                self.Liquidate(self.contract)
                self._record_trade(True)
                self.Debug(f"TAKE PROFIT HIT: {price:.2f}")
                return

            # Trailing stop (move stop to breakeven after 1.5 ATR profit)
            if price > self.entry_price + (1.5 * self.atr.Current.Value):
                self.stop_loss = max(self.stop_loss, self.entry_price)

        elif position.IsShort:
            # Check stop loss
            if price >= self.stop_loss:
                self.Liquidate(self.contract)
                self._record_trade(False)
                self.Debug(f"STOP LOSS HIT: {price:.2f}")
                return

            # Check take profit
            if price <= self.take_profit:
                self.Liquidate(self.contract)
                self._record_trade(True)
                self.Debug(f"TAKE PROFIT HIT: {price:.2f}")
                return

            # Trailing stop
            if price < self.entry_price - (1.5 * self.atr.Current.Value):
                self.stop_loss = min(self.stop_loss, self.entry_price)

    def _record_trade(self, is_winner):
        """Track trade statistics"""
        self.recent_trades.append(is_winner)

        if is_winner:
            self.win_streak += 1
            self.loss_streak = 0
        else:
            self.loss_streak += 1
            self.win_streak = 0

        # Calculate win rate
        if len(self.recent_trades) >= 10:
            win_rate = sum(self.recent_trades) / len(self.recent_trades)
            self.Debug(f"Win Rate (last {len(self.recent_trades)}): {win_rate:.1%}")

    def OnMarketOpen(self):
        """Reset daily metrics"""
        self.daily_start_equity = self.Portfolio.TotalPortfolioValue
        self.session_high_equity = self.Portfolio.TotalPortfolioValue
        self.daily_pnl = 0

        # Reduce position sizing after losing days
        yesterday_pnl = self.Portfolio.TotalPortfolioValue - self.daily_start_equity
        if yesterday_pnl < 0:
            self.max_contracts = max(1, self.max_contracts - 1)
        elif self.win_streak >= 3:
            self.max_contracts = min(2, self.max_contracts + 1)

    def BeforeMarketClose(self):
        """Flatten positions before close (day trading rule)"""
        if self.Portfolio.Invested:
            self.Liquidate()
            self.Debug("CLOSING: Flattening EOD position")

    def OnEndOfAlgorithm(self):
        """Final statistics"""
        total_return = (self.Portfolio.TotalPortfolioValue / 50000 - 1) * 100
        win_rate = sum(self.recent_trades) / len(self.recent_trades) if self.recent_trades else 0

        self.Debug("=" * 50)
        self.Debug(f"FINAL RESULTS:")
        self.Debug(f"Total Return: {total_return:.2f}%")
        self.Debug(f"Win Rate: {win_rate:.1%}")
        self.Debug(f"Total Trades: {len(self.recent_trades)}")
        self.Debug("=" * 50)