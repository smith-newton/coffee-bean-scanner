import streamlit as st
import requests

st.title("☕ Coffee Bean Scanner")

uploaded_file = st.file_uploader("Upload Coffee Bean Image", type=["jpg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        files = {"file": uploaded_file.getvalue()}

        response = requests.post("http://127.0.0.1:8000/predict", files=files)

        result = response.json()

        st.success(f"Class: {result['class']}")
        st.write(f"Confidence: {result['confidence']:.2f}")

        # Display recommendations
        st.subheader("🍵 Recommended Drinks")
        recommendations = result.get("recommendations", {})
        if recommendations:
            st.write(f"**{recommendations.get('description', '')}**")

            drinks = recommendations.get("drinks", [])
            if drinks:
                st.write("**Perfect for:**")
                for drink in drinks:
                    st.write(f"• {drink}")
        else:
            st.write("No recommendations available.")