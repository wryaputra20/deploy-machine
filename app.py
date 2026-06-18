{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPo9ey7MhJM/uieEWzJxII3"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 11,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "collapsed": true,
        "id": "mBv89Zy2B0rF",
        "outputId": "c374e82f-5a64-40c7-bae3-62d6471cddf9"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2026-06-18 04:46:43.462 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.465 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.466 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.469 No runtime found, using MemoryCacheStorageManager\n",
            "2026-06-18 04:46:43.480 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.481 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.483 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.486 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.487 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.488 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.490 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.491 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.492 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.495 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.496 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.497 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.499 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.500 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.502 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.504 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.506 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.508 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.510 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.512 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.513 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.517 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.518 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.519 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.522 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.523 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.524 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.528 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.530 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.531 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.532 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.535 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.536 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.538 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.540 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2026-06-18 04:46:43.541 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ],
      "source": [
        "import streamlit as st\n",
        "import pandas as pd\n",
        "from sklearn.datasets import load_iris\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "\n",
        "st.title(\"🌸 Iris Flower Classification App\")\n",
        "\n",
        "# Load and cache data\n",
        "@st.cache_data\n",
        "def load_data():\n",
        "    iris = load_iris()\n",
        "    X = pd.DataFrame(iris.data, columns=iris.feature_names)\n",
        "    y = pd.Series(iris.target)\n",
        "    return X, y, iris.target_names\n",
        "\n",
        "X, y, target_names = load_data()\n",
        "\n",
        "# Train and cache model\n",
        "@st.cache_resource\n",
        "def train_model(X, y):\n",
        "    model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
        "    model.fit(X, y)\n",
        "    return model\n",
        "\n",
        "model = train_model(X, y)\n",
        "\n",
        "# Sidebar inputs\n",
        "st.sidebar.header(\"Input Features\")\n",
        "sepal_length = st.sidebar.slider(\"Sepal length (cm)\", float(X.iloc[:,0].min()), float(X.iloc[:,0].max()), float(X.iloc[:,0].mean()))\n",
        "sepal_width  = st.sidebar.slider(\"Sepal width (cm)\",  float(X.iloc[:,1].min()), float(X.iloc[:,1].max()), float(X.iloc[:,1].mean()))\n",
        "petal_length = st.sidebar.slider(\"Petal length (cm)\", float(X.iloc[:,2].min()), float(X.iloc[:,2].max()), float(X.iloc[:,2].mean()))\n",
        "petal_width  = st.sidebar.slider(\"Petal width (cm)\",  float(X.iloc[:,3].min()), float(X.iloc[:,3].max()), float(X.iloc[:,3].mean()))\n",
        "\n",
        "input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=X.columns)\n",
        "\n",
        "# Predict\n",
        "if st.button(\"Predict\"):\n",
        "    prediction = model.predict(input_data)[0]\n",
        "    proba = model.predict_proba(input_data)[0]\n",
        "\n",
        "    st.subheader(\"Prediction Result\")\n",
        "    st.success(f\"🌼 Species: **{target_names[prediction]}**\")\n",
        "\n",
        "    st.subheader(\"Prediction Probabilities\")\n",
        "    st.bar_chart(pd.DataFrame(proba, index=target_names, columns=[\"Probability\"]))\n",
        "else:\n",
        "    st.info(\"⬅️ Masukkan fitur dan klik Predict\")"
      ]
    }
  ]
}