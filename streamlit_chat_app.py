import streamlit as st
import requests
import json
import time

# Configure Streamlit page
st.set_page_config(
    page_title="ASAI Chat Bot",
    page_icon="🍳",
    layout="centered"
)

# Enhanced CSS with better contrast and accessibility
st.markdown("""
<style>
    /* Custom color scheme with high contrast */
    .main {
        background-color: #fafafa;
    }
    
    .user-message {
        background-color: #ffffff;
        color: #1a1a1a;
        padding: 1.2rem;
        border-radius: 0.75rem;
        margin: 0.75rem 0;
        border-left: 4px solid #1976d2;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-size: 1rem;
        line-height: 1.5;
    }
    
    .bot-message {
        background-color: #f8f9fa;
        color: #212529;
        padding: 1.2rem;
        border-radius: 0.75rem;
        margin: 0.75rem 0;
        border-left: 4px solid #28a745;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-size: 1rem;
        line-height: 1.5;
    }
    
    .product-card {
        background-color: #ffffff;
        border: 2px solid #17a2b8;
        border-radius: 0.75rem;
        padding: 1.25rem;
        margin: 0.75rem 0;
        box-shadow: 0 3px 6px rgba(0,0,0,0.12);
        transition: transform 0.2s ease;
    }
    
    .product-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 12px rgba(0,0,0,0.15);
    }
    
    .product-name {
        color: #495057;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .product-price {
        color: #dc3545;
        font-size: 1.2rem;
        font-weight: 700;
    }
    
    .product-description {
        color: #6c757d;
        font-style: italic;
        margin-top: 0.5rem;
    }
    
    .product-reviews {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #e9ecef;
    }
    
    .reviews-header {
        color: #495057;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .review-item {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #ffc107;
    }
    
    .review-header {
        display: flex;
        align-items: center;
        margin-bottom: 0.25rem;
    }
    
    .review-rating {
        color: #ffc107;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    
    .review-author {
        color: #495057;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .review-verified {
        color: #28a745;
        font-size: 0.75rem;
        margin-left: 0.5rem;
    }
    
    .review-text {
        color: #6c757d;
        font-size: 0.9rem;
        line-height: 1.4;
        font-style: italic;
    }
    
    .streaming-text {
        color: #212529;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    .message-label {
        font-weight: 600;
        color: #495057;
    }
    
    /* Better button styling */
    .stButton > button {
        background-color: #28a745;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: background-color 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #218838;
    }
    
    /* Chat input styling */
    .stTextInput > div > div > input {
        border-radius: 0.5rem;
        border: 2px solid #dee2e6;
        font-size: 1rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #1976d2;
        box-shadow: 0 0 0 0.2rem rgba(25, 118, 210, 0.25);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .user-message, .bot-message, .product-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# API URL (updated to use modular backend)
API_URL = "http://127.0.0.1:5001"

def format_reviews_html(reviews):
    """Format product reviews as HTML"""
    if not reviews:
        return ""
    
    reviews_html = '<div class="product-reviews">'
    reviews_html += '<div class="reviews-header">👥 Customer Reviews</div>'
    
    for review in reviews[:2]:  # Show max 2 reviews to keep cards manageable
        stars = "⭐" * review.get('rating', 0)
        verified_badge = ' <span class="review-verified">✓ Verified</span>' if review.get('verified', False) else ''
        
        reviews_html += f'''
        <div class="review-item">
            <div class="review-header">
                <span class="review-rating">{stars}</span>
                <span class="review-author">{review.get('author', 'Anonymous')}</span>
                {verified_badge}
            </div>
            <div class="review-text">"{review.get('text', '')}"</div>
        </div>
        '''
    
    reviews_html += '</div>'
    return reviews_html

def stream_response_to_ui(messages):
    """Stream response from API and display in real-time"""
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"messages": messages},
            stream=True,
            timeout=30
        )
        
        if response.status_code == 200:
            # Create containers for streaming
            response_container = st.empty()
            current_text = ""
            recommended_products = []
            
            # Process streaming response
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    data_content = line[6:]
                    
                    if data_content.strip() == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(data_content)
                        
                        # Handle text chunks
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                current_text += content
                                # Update the display in real-time
                                response_container.markdown(f"""
                                <div class="bot-message">
                                    <span class="message-label">ASAI-Bot:</span>
                                    <div class="streaming-text">{current_text}</div>
                                </div>
                                """, unsafe_allow_html=True)
                                time.sleep(0.01)  # Small delay for smoother streaming effect
                        
                        # Handle product chunks
                        elif chunk.get("object") == "chat.completion.products":
                            recommended_products = chunk.get("products", [])
                            
                    except json.JSONDecodeError:
                        continue
            
            return current_text, recommended_products
        else:
            st.error(f"❌ API Error: {response.status_code}")
            return None, []
            
    except Exception as e:
        st.error(f"❌ Connection Error: {e}")
        return None, []

def send_message_to_api(messages):
    """Fallback non-streaming method"""
    try:
        response = requests.post(
            f"{API_URL}/ask",
            json={"messages": messages},
            stream=True,
            timeout=30
        )
        
        if response.status_code == 200:
            full_response = ""
            recommended_products = []
            
            # Process streaming response
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    data_content = line[6:]
                    
                    if data_content.strip() == "[DONE]":
                        break
                    
                    try:
                        chunk = json.loads(data_content)
                        
                        # Handle text chunks
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                full_response += content
                        
                        # Handle product chunks
                        elif chunk.get("object") == "chat.completion.products":
                            recommended_products = chunk.get("products", [])
                            
                    except json.JSONDecodeError:
                        continue
            
            return full_response, recommended_products
        else:
            st.error(f"❌ API Error: {response.status_code}")
            return None, []
            
    except Exception as e:
        st.error(f"❌ Connection Error: {e}")
        return None, []

# Main app
st.title("🍳 ASAI Cookware Chat Bot")
st.markdown("**Ask me anything about ASAI ceramic cookware!** I'll help you find the perfect products for your kitchen. ✨")

# Add some spacing
st.markdown("<br>", unsafe_allow_html=True)

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="user-message">
                <span class="message-label">You:</span> {message["content"]}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="bot-message">
                <span class="message-label">ASAI-Bot:</span>
                <div class="streaming-text">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show products if available
            if "products" in message and message["products"]:
                st.markdown("### 🛍️ Recommended Products")
                
                # Create columns for better layout on wider screens
                cols = st.columns(1)
                for idx, product in enumerate(message["products"]):
                    with cols[idx % len(cols)]:
                        reviews_html = format_reviews_html(product.get('reviews', []))
                        st.markdown(f"""
                        <div class="product-card">
                            <div class="product-name">{product.get('name', 'Unknown Product')}</div>
                            <div class="product-price">{product.get('price', 'N/A')} - {product.get('size', 'N/A')}</div>
                            <div class="product-description">{product.get('description', 'No description available')}</div>
                            {reviews_html}
                        </div>
                        """, unsafe_allow_html=True)

# Add spacing before input
st.markdown("<br>", unsafe_allow_html=True)

# Chat input section
st.markdown("### 💬 Send a Message")

# Use form for better UX (Enter key support)
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message:", 
            placeholder="e.g., 'Show me non-stick pans' or 'What's best for healthy cooking?'",
            label_visibility="collapsed",
            key="user_input_field"
        )
    
    with col2:
        send_button = st.form_submit_button("Send 📤", use_container_width=True)

# Handle sending message
if send_button and user_input.strip():
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Show the user message immediately
    st.markdown(f"""
    <div class="user-message">
        <span class="message-label">You:</span> {user_input}
    </div>
    """, unsafe_allow_html=True)
    
    # Create a status indicator
    with st.spinner("🤖 ASAI-Bot is thinking..."):
        # Stream the response
        response_text, products = stream_response_to_ui(st.session_state.messages)
    
    if response_text:
        # Add bot response to session state
        bot_message = {"role": "assistant", "content": response_text}
        if products:
            bot_message["products"] = products
        st.session_state.messages.append(bot_message)
        
        # Show products if available
        if products:
            st.markdown("### 🛍️ Recommended Products")
            for product in products:
                reviews_html = format_reviews_html(product.get('reviews', []))
                st.markdown(f"""
                <div class="product-card">
                    <div class="product-name">{product.get('name', 'Unknown Product')}</div>
                    <div class="product-price">{product.get('price', 'N/A')} - {product.get('size', 'N/A')}</div>
                    <div class="product-description">{product.get('description', 'No description available')}</div>
                    {reviews_html}
                </div>
                """, unsafe_allow_html=True)
        
        # Refresh to show the complete conversation
        st.rerun()

# Control buttons
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6c757d; font-size: 0.9rem;'>"
    "Powered by ASAI Cookware • Built with ❤️ using Streamlit"
    "</div>", 
    unsafe_allow_html=True
)
