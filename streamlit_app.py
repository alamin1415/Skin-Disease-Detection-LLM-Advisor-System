import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="Skin Disease AI Dashboard", layout="wide")

# ---------------- Header ----------------
st.markdown(
    """
    <h2 style='text-align:center;'>Skin Disease AI Dashboard</h2>
    <p style='text-align:center;color:gray;'>
        AI-powered skin disease detection system
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- Session ----------------
if "result" not in st.session_state:
    st.session_state.result = None

# ---------------- Layout ----------------
left, right = st.columns([1, 1], gap="large")

# ================= LEFT SIDE =================
with left:
    st.subheader("Upload Panel")

    uploaded_file = st.file_uploader(
        "Upload image (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"]
    )

    image_box = st.container()
    action_box = st.container()

    if uploaded_file:
        with image_box:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        with action_box:
            st.write("")  # spacing consistency

            if st.button("Run Analysis", use_container_width=True):

                with st.spinner("Analyzing..."):

                    try:
                        response = requests.post(
                            "http://127.0.0.1:8000/analyze_skin",
                            files={
                                "file": (
                                    uploaded_file.name,
                                    uploaded_file.getvalue(),
                                    uploaded_file.type
                                )
                            },
                            timeout=60
                        )

                        if response.status_code == 200:
                            st.session_state.result = response.json()
                        else:
                            st.error("Server error")
                            st.write(response.text)

                    except Exception:
                        st.error("Backend connection failed")

# ================= RIGHT SIDE =================
with right:
    st.subheader("Result Panel")

    result_box = st.container()

    with result_box:
        result = st.session_state.result

        if result:
            st.metric("Disease", result.get("disease", "N/A"))
            st.metric("Confidence", f"{result.get('confidence', 0):.2f}")

            st.markdown("---")

            st.markdown("### Recommendations")
            st.write(result.get("recommendations", "N/A"))

            st.markdown("### Next Steps")
            st.write(result.get("next_steps", "N/A"))

        else:
            st.info("No result yet. Upload an image and run analysis.")