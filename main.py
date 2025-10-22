import json
import os
import io
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from openai import OpenAI

# -------------------- SETUP --------------------
# Load API key safely
openai_api_key = os.getenv("OPENAI_API_KEY") or open("API_KEY", "r").read().strip()
client = OpenAI(api_key=openai_api_key)

st.set_page_config(page_title="Stock Analysis Chatbot", page_icon="📈")

st.title("📊 Stock Analysis Chatbot Assistant")

# -------------------- STOCK FUNCTIONS --------------------

def get_stock_price(ticker: str) -> float:
    """Get the latest closing stock price for the ticker."""
    data = yf.Ticker(ticker).history(period="1y")
    return round(float(data.iloc[-1].Close), 2)

def calculate_SMA(ticker: str, window: int) -> float:
    """Calculate the Simple Moving Average."""
    data = yf.Ticker(ticker).history(period="1y").Close
    return round(float(data.rolling(window=window).mean().iloc[-1]), 2)

def calculate_EMA(ticker: str, window: int) -> float:
    """Calculate the Exponential Moving Average."""
    data = yf.Ticker(ticker).history(period="1y").Close
    return round(float(data.ewm(span=window, adjust=False).mean().iloc[-1]), 2)

def calculate_RSI(ticker: str) -> float:
    """Calculate the Relative Strength Index."""
    data = yf.Ticker(ticker).history(period="1y").Close
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ema_up = up.ewm(com=14 - 1, adjust=False).mean()
    ema_down = down.ewm(com=14 - 1, adjust=False).mean()
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2)

def calculate_MACD(ticker: str):
    """Calculate MACD, Signal, and Histogram."""
    data = yf.Ticker(ticker).history(period="1y").Close
    short_EMA = data.ewm(span=12, adjust=False).mean()
    long_EMA = data.ewm(span=26, adjust=False).mean()
    MACD = short_EMA - long_EMA
    signal = MACD.ewm(span=9, adjust=False).mean()
    hist = MACD - signal
    return {
        "MACD": round(float(MACD.iloc[-1]), 2),
        "Signal": round(float(signal.iloc[-1]), 2),
        "Histogram": round(float(hist.iloc[-1]), 2)
    }

def plot_stock_price(ticker: str):
    """Plot last 1-year stock price and return image buffer."""
    data = yf.Ticker(ticker).history(period="1y")
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data.Close, label="Close Price", color="blue")
    plt.title(f"{ticker} Stock Price Over Last Year")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.grid(True)
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close()
    return buf

# -------------------- FUNCTION DEFINITIONS FOR GPT --------------------

functions = [
    {
        "name": "get_stock_price",
        "description": "Gets the latest stock price given the ticker symbol of a company.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., AAPL)."}
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "calculate_SMA",
        "description": "Calculates the Simple Moving Average for a stock and window.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "window": {"type": "integer"}
            },
            "required": ["ticker", "window"],
        },
    },
    {
        "name": "calculate_EMA",
        "description": "Calculates the Exponential Moving Average for a stock and window.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "window": {"type": "integer"}
            },
            "required": ["ticker", "window"],
        },
    },
    {
        "name": "calculate_RSI",
        "description": "Calculates the Relative Strength Index for a stock.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"}
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "calculate_MACD",
        "description": "Calculates the MACD, Signal Line, and Histogram for a stock.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"}
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "plot_stock_price",
        "description": "Plots the stock price for the last year.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"}
            },
            "required": ["ticker"],
        },
    },
]

available_functions = {
    "get_stock_price": get_stock_price,
    "calculate_SMA": calculate_SMA,
    "calculate_EMA": calculate_EMA,
    "calculate_RSI": calculate_RSI,
    "calculate_MACD": calculate_MACD,
    "plot_stock_price": plot_stock_price,
}

# -------------------- STREAMLIT APP LOGIC --------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = []

user_input = st.text_input("💬 Your input:")

if user_input:
    try:
        st.session_state["messages"].append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=st.session_state["messages"],
            functions=functions,
            function_call="auto",
        )

        response_message = response.choices[0].message

        # Check if the model wants to call a function
        if response_message.function_call is not None:
            function_name = response_message.function_call.name
            function_args = json.loads(response_message.function_call.arguments)
            function_to_call = available_functions.get(function_name)

            if not function_to_call:
                st.error(f"Unknown function: {function_name}")
            else:
                result = function_to_call(**function_args)

                if function_name == "plot_stock_price":
                    st.image(result)
                else:
                    # Send result back to model for final response
                    st.session_state["messages"].append(
                        {"role": "function", "name": function_name, "content": json.dumps(result)}
                    )

                    second_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=st.session_state["messages"]
                    )
                    final_msg = second_response.choices[0].message.content
                    st.write(final_msg)
                    st.session_state["messages"].append({"role": "assistant", "content": final_msg})
        else:
            # Direct answer from model
            st.write(response_message.content)
            st.session_state["messages"].append({"role": "assistant", "content": response_message.content})

    except Exception as e:
        import traceback
        st.error(f"❌ Error occurred: {e}")
        st.text(traceback.format_exc())
