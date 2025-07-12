# 🤖 AI-Powered Multi-Agent Trading System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-412991?logo=openai&logoColor=white)](https://openai.com/)

An advanced AI-powered trading system that combines **multi-agent coordination**, **sentiment analysis**, and **technical analysis** for sophisticated cryptocurrency trading decisions. Built with LangChain, Streamlit, and integrated with real-time market data.

![Trading System Demo](https://img.shields.io/badge/Demo-Live-green) 

## ✨ Key Features

### 🚀 **Advanced AI Trading**
- **Multi-Agent Architecture**: Coordinated agents for technical analysis, sentiment analysis, and risk management
- **Real-Time Sentiment Analysis**: Analyzes social media posts from influential accounts (Elon Musk, Donald Trump, etc.)
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages with professional TradingView-style charts
- **Smart Decision Making**: LangChain-powered decision engine with reasoning transparency

### 📊 **Professional Trading Interface**
- **Interactive Dashboard**: Streamlit-based web interface with real-time updates
- **Trade History & P&L**: Comprehensive trade tracking with profit/loss analysis and sentiment correlation
- **Performance Metrics**: Win rate, drawdown analysis, portfolio performance vs buy-and-hold
- **Technical Charts**: Professional candlestick charts with trading signals and indicators

### 🎯 **Smart Features**
- **Sentiment-Driven Trading**: Correlates social media sentiment with trading decisions
- **Risk Management**: Configurable position sizing and stop-loss mechanisms
- **Historical Analysis**: Review past simulation results and trading performance
- **Real-Time Reasoning**: Transparent agent decision-making process

### 🛠 **Technical Architecture**
- **LangChain Framework**: Structured agent orchestration and tool management
- **Real-Time Data**: Live cryptocurrency data via CCXT and Binance API
- **Advanced Analytics**: Technical analysis with TA-Lib integration
- **Secure Configuration**: Environment-based API key management

## 🎬 Demo Screenshots

### Trading Dashboard
![Trading Dashboard](docs/dashboard.png)

### Sentiment Analysis
![Sentiment Analysis](docs/sentiment.png)

### Performance Analytics
![Performance Analytics](docs/performance.png)

## 📋 Prerequisites

- **Python 3.8+** (Recommended: Python 3.9 or 3.10)
- **OpenAI API Key** (for AI decision-making agent)
- **Internet Connection** (for real-time market data)
- **Optional**: PostgreSQL database (for persistent result storage)

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-trading-system.git
cd ai-trading-system
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv trading-env
source trading-env/bin/activate  # On Windows: trading-env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Create environment file
cp .env.example .env

# Edit with your settings
nano .env
```

**Required environment variables:**
```env
OPENAI_API_KEY=your_openai_api_key_here
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_db
DB_USER=your_db_user
DB_PASSWORD=your_db_password
```

### 4. Run the Application

```bash
streamlit run auto-trade.py
```

Navigate to `http://localhost:8501` in your browser to access the trading interface.

### 5. Run Tests (Optional)

```bash
# Quick validation test
python test/test_quick_validation.py

# Comprehensive test suite
python test/test_runner.py
```

## 🏗 Architecture Overview

### Multi-Agent System Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AI Trading System Architecture                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐  │
│  │   Technical     │  │   Sentiment     │  │    Trading          │  │
│  │   Analysis      │  │   Analysis      │  │    Decision         │  │
│  │   Agent         │  │   Agent         │  │    Agent            │  │
│  │   (Local)       │  │   (Local)       │  │    (OpenAI)         │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────┘  │
│           │                     │                     │              │
│           └─────────────────────┼─────────────────────┘              │
│                                 │                                    │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │              Streamlit Web Interface                            │ │
│  │  • Real-time Charts    • Trade History    • Performance       │ │
│  │  • Sentiment Display   • P&L Analysis     • Agent Reasoning   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                     Data Layer                                  │ │
│  │  • CCXT (Market Data)  • PostgreSQL (Storage)  • FAISS (ML)   │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Technical Analysis Agent** 🔍
   - RSI, MACD, Bollinger Bands calculation
   - Moving averages and trend analysis
## 📱 Usage Guide

### Starting a Trading Simulation

1. **Configure Parameters**
   - Set initial capital (default: $10,000)
   - Choose time period for backtesting
   - Select cryptocurrency pair (e.g., BTC/USDT)
   - Configure trading fees

2. **Run Simulation**
   - Click "Start Trading Simulation"
   - Monitor real-time agent decisions
   - View sentiment analysis and technical indicators
   - Track portfolio performance

3. **Analyze Results**
   - Review trade history with P&L analysis
   - Compare performance vs buy-and-hold
   - Examine sentiment correlation with trades
   - Export results for further analysis

### Key Interface Features

#### 🔮 **Next Action Recommendation**
- **Multi-agent voting system** for trading decisions
- **Technical Analysis Agent**: RSI, MACD, moving averages
- **Sentiment Analysis Agent**: Social media sentiment scoring
- **Risk Management Agent**: Position sizing and risk assessment
- **Final recommendation** with confidence scores

#### 📊 **Technical Analysis Dashboard**
- **Professional TradingView-style charts**
- **Candlestick patterns** with volume analysis
- **Technical indicators**: Bollinger Bands, RSI, MACD
- **Buy/sell signals** overlaid on price charts

#### 🎭 **Sentiment Analysis**
- **Real-time social media monitoring**
- **Influential account tracking** (Elon Musk, Donald Trump, etc.)
- **Sentiment scoring** and market impact assessment
- **Correlation with trading decisions**

#### 📈 **Performance Analytics**
- **Portfolio value tracking** over time
- **Win rate and profit metrics**
- **Drawdown analysis**
- **Comparison with market benchmarks**

## ⚙️ Configuration Options

### Trading Parameters

```python
# In the Streamlit interface, configure:
INITIAL_CAPITAL = 10000      # Starting portfolio value
BUY_FEE_PCT = 0.1           # Buy transaction fee (0.1%)
SELL_FEE_PCT = 0.1          # Sell transaction fee (0.1%)
MAX_POSITION_SIZE = 0.3     # Maximum 30% of portfolio per trade
STOP_LOSS_PCT = 0.05        # 5% stop-loss threshold
```

### Environment Variables

```env
# API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini  # or gpt-4, gpt-3.5-turbo

# Database Configuration (Optional)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_results
DB_USER=postgres
DB_PASSWORD=your_password

# Trading Configuration
DEFAULT_SYMBOL=BTC/USDT
DEFAULT_TIMEFRAME=1h
SENTIMENT_THRESHOLD=0.2
RISK_TOLERANCE=medium
```

## 🧪 Testing

### Quick Validation

Run the quick validation test to ensure everything is working:

```bash
python test/test_quick_validation.py
```

### Comprehensive Test Suite

For thorough testing of all components:

```bash
python test/test_runner.py
```

### Test Coverage

- ✅ Package imports and dependencies
- ✅ Main module and agent initialization  
- ✅ Data fetching and technical analysis
- ✅ Sentiment analysis functionality
- ✅ Trading simulation logic
- ✅ P&L calculation accuracy
- ✅ Display functions and UI components

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

### Key Performance Indicators
- **Total Return**: Overall strategy performance vs initial capital
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Average Trade Duration**: Typical holding period
- **Profit Factor**: Ratio of gross profit to gross loss

### Comparative Analysis
- **Strategy vs Buy & Hold**: Performance comparison
- **Risk Metrics**: Volatility and risk analysis
- **Trade Distribution**: Win/loss breakdown
- **Sentiment Correlation**: Impact of social media sentiment

## 🚀 Deployment

### Local Development

```bash
# Development mode with hot reload
streamlit run auto-trade.py --server.runOnSave true
```

### Production Deployment

#### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "auto-trade.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build and run
docker build -t ai-trading-system .
docker run -p 8501:8501 --env-file .env ai-trading-system
```

#### Cloud Deployment (Streamlit Cloud)

1. Push code to GitHub repository
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Configure environment variables in Streamlit Cloud dashboard
4. Deploy with one click

### Environment Security

- **Never commit API keys** to version control
- **Use environment variables** for all sensitive data
- **Enable rate limiting** for production deployments
- **Monitor API usage** and costs

## 🛠 Development

### Project Structure

```
ai-trading-system/
├── auto-trade.py              # Main application
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── .env.example             # Environment template
├── test/                    # Test suite
│   ├── test_runner.py       # Comprehensive tests
│   ├── test_quick_validation.py  # Quick validation
│   └── *.py                 # Individual test files
├── docs/                    # Documentation
└── .gitignore              # Git ignore rules
```

### Adding New Features

1. **Create feature branch**: `git checkout -b feature/your-feature`
2. **Follow coding standards**: Use type hints and docstrings
3. **Add tests**: Update test suite for new functionality
4. **Update documentation**: Modify README and code comments
5. **Submit pull request**: Include description and test results

### Code Style

- **PEP 8 compliance** for Python code formatting
- **Type hints** for all function parameters and returns
- **Comprehensive docstrings** for classes and methods
- **Error handling** with try/catch blocks
- **Logging** for debugging and monitoring

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Add features, fix bugs, improve documentation
4. **Add tests**: Ensure your changes are tested
5. **Run test suite**: `python test/test_runner.py`
6. **Commit changes**: `git commit -m 'Add amazing feature'`
7. **Push to branch**: `git push origin feature/amazing-feature`
8. **Open Pull Request**: Describe your changes and provide test results

### Contribution Areas

- 🔧 **Core Features**: Trading algorithms, agent improvements
- 📊 **Analytics**: New performance metrics, visualization enhancements
- 🎨 **UI/UX**: Streamlit interface improvements
- 📚 **Documentation**: README, code comments, tutorials
- 🧪 **Testing**: Additional test cases, performance testing
- 🔒 **Security**: Security improvements, code reviews

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/ai-trading-system.git
cd ai-trading-system

# Create development environment
python -m venv dev-env
source dev-env/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests before making changes
python test/test_runner.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**This is educational software for learning about algorithmic trading and AI systems.**

- **Not Financial Advice**: This system is for educational and research purposes only
- **No Investment Recommendations**: Do not use for actual trading without proper due diligence
- **Risk Warning**: Cryptocurrency trading involves substantial risk of loss
- **Use at Your Own Risk**: Authors are not responsible for any financial losses

## 📞 Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/yourusername/ai-trading-system/issues)
- **Discussions**: Join the conversation in [GitHub Discussions](https://github.com/yourusername/ai-trading-system/discussions)
- **Documentation**: Check the [Wiki](https://github.com/yourusername/ai-trading-system/wiki) for detailed guides

## 🙏 Acknowledgments

- **OpenAI** for GPT API and language models
- **LangChain** for agent framework and tools
- **Streamlit** for the excellent web framework
- **CCXT** for cryptocurrency exchange connectivity
- **TA-Lib** for technical analysis indicators
- **Plotly** for interactive charting capabilities

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**
│           └─────────────────────┼───────────────────┘       │
│                                 │                           │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │          Trading Decision Agent (OpenAI)                │  │
│  │         - Synthesizes all agent inputs                 │  │
│  │         - Makes final BUY/SELL/HOLD decisions          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                 │                           │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ Deep Learning   │  │ Vector Database │                   │
│  │ Agent           │  │ Agent           │                   │
│  │ (TensorFlow)    │  │ (FAISS)         │                   │
│  └─────────────────┘  └─────────────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

1. **Fast Local Agents** (No OpenAI calls)
   - Market Analyst: Technical indicator analysis
   - Pattern Recognition: Rule-based chart patterns
   - Risk Management: Position sizing and risk metrics

2. **AI Decision Maker** (OpenAI-powered)
   - Synthesizes all agent inputs
   - Makes final trading decisions
   - Provides detailed reasoning

3. **Machine Learning Components**
   - Deep Learning Agent: Neural network for pattern prediction
   - Vector Database: Historical pattern storage and retrieval

## 🎛 Configuration Options

### Trading Parameters

```python
config = TradingConfig(
    initial_capital=1000.0,      # Starting capital
    buy_fee_pct=0.10,            # 10% fee on purchases
    sell_fee_pct=0.0,            # No fee on sales
    enable_deep_learning=True,    # Enable ML predictions
    enable_vector_db=True,        # Enable pattern storage
    show_reasoning=True           # Show agent reasoning panel
)
```

### Advanced Settings

- **RSI Thresholds**: Oversold/overbought levels
- **Stop Loss/Take Profit**: Risk management parameters
- **Model Training**: Deep learning configuration
- **Vector Database**: Pattern similarity settings

## 📊 Performance Metrics

The system tracks comprehensive performance metrics:

### Key Metrics
- **Total Return**: Overall strategy performance
- **Win Rate**: Percentage of profitable trades
- **Max Drawdown**: Largest peak-to-trough decline
- **Trade Statistics**: Count, frequency, and success rate

### Real-time Monitoring
- Live portfolio value tracking
- Agent reasoning display
- Trade execution visualization
- Performance comparison charts

## 🔍 Agent Reasoning Panel

The right-side panel shows real-time agent decision-making:

- **📈 Market Analysis**: Technical indicator insights
- **🔍 Pattern Recognition**: Identified chart patterns
- **⚠️ Risk Assessment**: Current risk exposure
- **🤖 ML Prediction**: Deep learning recommendations
- **🔍 Historical Insights**: Vector database suggestions
- **🎯 Final Decision**: Synthesized trading decision

## 📚 Historical Results

View and analyze past trading results:

- **Results Table**: Summary of all past simulations
- **Detailed Analysis**: Deep dive into specific runs
- **Performance Comparison**: Compare different strategies
- **Trade Log Review**: Examine individual trade decisions

## 🛡 Security Features

- **Environment Variables**: Secure API key storage
- **Database Encryption**: Protected data storage
- **Error Handling**: Comprehensive error management
- **Rate Limiting**: API call optimization

## 🔧 Development

### Project Structure

```
Neural-Symbolic/
├── auto-trade.py           # Main application
├── requirements.txt        # Dependencies
├── .env.template          # Environment template
├── README.md             # Documentation
└── .gitignore           # Git ignore rules
```

### Adding New Agents

1. Create agent class inheriting from base structure
2. Implement required methods (`analyze`, `step`, etc.)
3. Add to `EnhancedMultiAgentTradingSystem`
4. Update UI components for new agent reasoning

### Extending ML Components

1. **Deep Learning**: Modify `DeepLearningAgent` class
2. **Vector Database**: Extend `VectorDBAgent` capabilities
3. **New Models**: Add additional prediction models
4. **Feature Engineering**: Enhance input features

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues and questions:
- Check the GitHub issues page
- Review the troubleshooting section
- Contact the development team

## 🚀 Roadmap

### Planned Features
- [ ] Additional technical indicators
- [ ] More sophisticated ML models
- [ ] Real-time market data integration
- [ ] Portfolio optimization algorithms
- [ ] Advanced risk management strategies
- [ ] Multi-asset trading support
- [ ] Cloud deployment options

### Performance Improvements
- [ ] Async processing for better speed
- [ ] Caching layer for repeated calculations
- [ ] Optimized database queries
- [ ] Real-time streaming updates

---

**⚠️ Disclaimer**: This system is for educational and research purposes. Past performance does not guarantee future results. Always conduct thorough testing before using with real funds.
- **Prompt engineering**: Specialized prompts for each agent type

### Technical Analysis
- **Multiple indicators**: SMA, Bollinger Bands, RSI, MACD
- **Pattern recognition**: Automated chart pattern detection
- **Risk metrics**: Real-time P&L and risk assessment
- **Portfolio tracking**: Complete portfolio state management

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set up environment variables**:
```bash
cp .env.template .env
# Edit .env with your actual values
```

3. **Set up PostgreSQL database**:
- Install PostgreSQL
- Create database `amaai_trading`
- Update database credentials in `.env`

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# OpenAI API Key for LangChain agents
OPENAI_API_KEY=your_openai_api_key_here

# Database configuration
DB_HOST=localhost
DB_PORT=5433
DB_NAME=amaai_trading
DB_USER=postgres
DB_PASSWORD=your_db_password_here

# Trading configuration
INITIAL_CAPITAL=1000.0
RSI_OVERSOLD=30.0
RSI_OVERBOUGHT=70.0
STOP_LOSS_PCT=0.05
TAKE_PROFIT_PCT=0.10
```

### Trading Parameters

Adjust trading parameters in the `TradingConfig` class:

```python
class TradingConfig(BaseModel):
    initial_capital: float = 1000.0
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    max_position_size: float = 1.0
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
```

## Usage

### Running the Application

```bash
streamlit run auto-trade.py
```

### Using the Interface

1. **Set parameters** in the sidebar:
   - Trading symbol (e.g., BTC/USDT)
   - Time interval (15m, 1h, 4h, 1d)
   - Start and end dates
   - Initial capital

2. **Run simulation**: Click "Run Simulation" to start the multi-agent trading system

3. **View results**:
   - Performance metrics
   - Interactive charts
   - Trade log with agent reasoning
   - Portfolio value tracking

## Multi-Agent Workflow

The system follows this workflow for each trading decision:

1. **Data Collection**: Fetch market data and calculate technical indicators
2. **Market Analysis**: MarketAnalystAgent analyzes current market conditions
3. **Pattern Recognition**: PatternRecognitionAgent identifies trading patterns
4. **Risk Assessment**: RiskManagementAgent evaluates current risk exposure
5. **Decision Making**: TradingDecisionAgent synthesizes all inputs
6. **Execution**: MultiAgentTradingSystem executes the trading decision
7. **Tracking**: Results are logged and portfolio is updated

## Database Schema

The system stores results in PostgreSQL with the following schema:

```sql
CREATE TABLE backtest_runs (
    id SERIAL PRIMARY KEY,
    run_timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    interval TEXT NOT NULL,
    initial_capital NUMERIC NOT NULL,
    strategy_return FLOAT,
    buyhold_return FLOAT,
    projection TEXT,
    trade_log JSONB,
    price_history JSONB,
    knowledge_graph JSONB
);
```

## Customization

### Adding New Agents

To add a new agent:

1. Create a new tool class inheriting from `BaseTool`
2. Implement the agent class with LangChain integration
3. Add the agent to `MultiAgentTradingSystem`
4. Update the decision-making process

### Modifying Trading Logic

The trading logic can be customized in:
- `TradingDecisionAgent.decide()`: Final decision logic
- `MultiAgentTradingSystem._execute_decision()`: Execution logic
- Individual agent analysis methods

### Adding New Indicators

To add new technical indicators:

1. Add indicator calculation in `fetch_binance_ta()`
2. Update `TechnicalAnalysisTool` to include new indicators
3. Modify agent prompts to consider new indicators

## Security Considerations

- **API Keys**: Never commit API keys to version control
- **Database**: Use strong passwords and proper network security
- **Input Validation**: All user inputs are validated
- **Error Handling**: Sensitive information is not exposed in error messages

## Performance Optimization

- **Caching**: Consider implementing caching for repeated calculations
- **Batch Processing**: API calls are batched for efficiency
- **Memory Management**: Large datasets are processed in chunks
- **Database Indexing**: Proper indexing for query performance

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure all dependencies are installed
2. **Database connection**: Check database credentials and connectivity
3. **API limits**: Monitor OpenAI API usage and rate limits
4. **Memory issues**: Reduce data range for large backtests

### Debugging

Enable debug logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with proper testing
4. Submit a pull request

## License

This project is licensed under the MIT License.
