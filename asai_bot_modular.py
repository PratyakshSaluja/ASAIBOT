import os
import json
import time
import uuid
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
from flask import Flask, request, Response
from dotenv import load_dotenv

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Load environment variables
load_dotenv()

# --- CUSTOM STREAMING CALLBACK ---

class StreamingCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []
        self.current_token = ""
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.current_token = token
        self.tokens.append(token)

# --- 1. ENUMS AND DATA CLASSES ---

class ConversationState(Enum):
    GREETING = "greeting"
    COOKING_INQUIRY = "cooking_inquiry"
    PRODUCT_DETAILS = "product_details"
    SIZE_COMPARISON = "size_comparison"
    GENERAL_CHAT = "general_chat"
    CLOSING = "closing"

class IntentType(Enum):
    GREETING = "greeting"
    COOKING_NEED = "cooking_need"
    PRODUCT_QUESTION = "product_question"
    SIZE_INQUIRY = "size_inquiry"
    GENERAL_INFO = "general_info"
    NEGATIVE_RESPONSE = "negative_response"
    AFFIRMATIVE_RESPONSE = "affirmative_response"
    CONVERSATION_HISTORY = "conversation_history"

@dataclass
class ConversationContext:
    state: ConversationState
    intent: IntentType
    mentioned_dishes: List[str]
    mentioned_sizes: List[str]
    user_preferences: Dict[str, Any]
    conversation_history: List[Dict[str, str]]

# --- 2. CONFIGURATION ---

app = Flask(__name__)

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

# Load product database
def load_product_database():
    try:
        with open('products.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Warning: products.json not found.")
        return {"products": []}
    except json.JSONDecodeError:
        print("Warning: Invalid JSON in products.json.")
        return {"products": []}

PRODUCT_DB = load_product_database()
PDF_PATH = "test.pdf"

# --- 3. INTENT CLASSIFIER ---

class IntentClassifier:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        self.intent_prompt = PromptTemplate(
            input_variables=["query"],
            template="""Classify the user's intent based on their message.

Current Message: {query}

Intent Categories:
- greeting: "hi", "hello", "hey"
- cooking_need: asking about cooking specific dishes
- product_question: asking about product details, features, prices
- size_inquiry: asking about sizes, dimensions
- general_info: asking about company, ceramic benefits
- negative_response: "no", "not interested", "not looking"
- affirmative_response: "yes", "tell me more", "I want"
- conversation_history: asking about previous questions/conversation

IMPORTANT: If the user says "yes", "tell me more", or similar affirmative responses, classify as "affirmative_response"

Return ONLY the intent category (one word):"""
        )
        
    def classify(self, query: str) -> IntentType:
        try:
            # Fast rule-based classification first
            quick_intent = self._quick_classify(query)
            if quick_intent:
                return quick_intent
            
            # Use LLM only if rule-based fails
            response = self.llm.invoke(
                self.intent_prompt.format(query=query)
            )
            intent_text = response.content.strip().lower()
            
            # Map to enum
            intent_mapping = {
                "greeting": IntentType.GREETING,
                "cooking_need": IntentType.COOKING_NEED,
                "product_question": IntentType.PRODUCT_QUESTION,
                "size_inquiry": IntentType.SIZE_INQUIRY,
                "general_info": IntentType.GENERAL_INFO,
                "negative_response": IntentType.NEGATIVE_RESPONSE,
                "affirmative_response": IntentType.AFFIRMATIVE_RESPONSE,
                "conversation_history": IntentType.CONVERSATION_HISTORY,
            }
            
            return intent_mapping.get(intent_text, IntentType.GENERAL_INFO)
        except:
            return IntentType.GENERAL_INFO
    
    def _quick_classify(self, query: str) -> Optional[IntentType]:
        """Fast rule-based classification to avoid LLM calls for obvious intents"""
        query_lower = query.lower().strip()
        
        # Greeting patterns
        greeting_patterns = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
        if any(pattern in query_lower for pattern in greeting_patterns):
            return IntentType.GREETING
            
        # Affirmative responses
        affirmative_patterns = ["yes", "yeah", "yep", "sure", "tell me more", "i want", "interested"]
        if any(pattern in query_lower for pattern in affirmative_patterns):
            return IntentType.AFFIRMATIVE_RESPONSE
            
        # Negative responses
        negative_patterns = ["no", "not interested", "don't want", "not looking"]
        if any(pattern in query_lower for pattern in negative_patterns):
            return IntentType.NEGATIVE_RESPONSE
            
        # Cooking needs
        cooking_patterns = ["cook", "cooking", "make", "making", "recipe", "dish", "meal"]
        if any(pattern in query_lower for pattern in cooking_patterns):
            return IntentType.COOKING_NEED
            
        # Size inquiries
        size_patterns = ["size", "dimension", "inches", "inch", "big", "small", "large"]
        if any(pattern in query_lower for pattern in size_patterns):
            return IntentType.SIZE_INQUIRY
            
        # Product questions
        product_patterns = ["price", "cost", "feature", "material", "ceramic", "pan", "pot", "cookware"]
        if any(pattern in query_lower for pattern in product_patterns):
            return IntentType.PRODUCT_QUESTION
            
        # Conversation history
        history_patterns = ["what did", "previous", "earlier", "before", "last question"]
        if any(pattern in query_lower for pattern in history_patterns):
            return IntentType.CONVERSATION_HISTORY
            
        return None

# --- 4. RESPONSE GENERATORS ---

class ResponseGenerator:
    def __init__(self, llm: ChatOpenAI, vector_store, product_db: Dict):
        self.llm = llm
        self.vector_store = vector_store
        self.product_db = product_db
        
        # Different prompts for different intents
        self.greeting_prompt = PromptTemplate(
            input_variables=["context", "conversation_history"],
            template="""You are ASAI-Bot. Give a brief, friendly greeting and ask how you can help with ceramic cookware needs.

Previous conversation:
{conversation_history}

Context: {context}

Keep it to 1-2 sentences max. End with a question."""
        )
        
        self.cooking_prompt = PromptTemplate(
            input_variables=["query", "context", "dish", "conversation_history"],
            template="""You are ASAI-Bot. The customer wants to cook {dish}.

Previous conversation:
{conversation_history}

Context from knowledge base:
{context}

Customer Query: {query}

IMPORTANT: ONLY use information from the context above. Explain briefly WHY ASAI ceramic cookware is perfect for {dish} based on the product information in the context. Do NOT provide recipes or cooking instructions not found in the context.

Keep response to 2-3 sentences. End with a relevant question about ASAI products."""
        )
        
        self.affirmative_response_prompt = PromptTemplate(
            input_variables=["context", "conversation_history"],
            template="""You are ASAI-Bot. The customer said "yes" or wants more information.

Previous conversation:
{conversation_history}

Context from knowledge base:
{context}

IMPORTANT: ONLY use information from the context above. Do NOT provide recipes, cooking instructions, or information not explicitly mentioned in the context. Based on the conversation history, provide the specific information they were asking for using ONLY the knowledge base context.

If the context doesn't contain the specific information they want, say "I don't have that specific information in my knowledge base, but I can tell you about [related ASAI information from context]."

Keep response to 2-3 sentences."""
        )
        
        self.product_prompt = PromptTemplate(
            input_variables=["query", "context", "conversation_history"],
            template="""You are ASAI-Bot. Answer the customer's product question using ONLY the context below.

Previous conversation:
{conversation_history}

Context from knowledge base:
{context}

Customer Query: {query}

IMPORTANT: ONLY use information explicitly found in the context above. Do NOT add external knowledge. If the specific information isn't in the context, say "I don't have that specific detail in my knowledge base."

Be specific but concise (2-3 sentences). End with a follow-up question."""
        )
        
        self.size_prompt = PromptTemplate(
            input_variables=["context", "conversation_history"],
            template="""You are ASAI-Bot. Explain ASAI's available sizes briefly.

Previous conversation:
{conversation_history}

Context: {context}

Available sizes: 8-inch (single/small portions), 10-inch (most popular), 12-inch (family size)
Keep to 2-3 sentences. End with asking about their preference."""
        )
        
        self.general_prompt = PromptTemplate(
            input_variables=["query", "context", "conversation_history"],
            template="""You are ASAI-Bot. Answer using ONLY the context provided below.

Previous conversation:
{conversation_history}

Context from knowledge base:
{context}

Customer Query: {query}

IMPORTANT: ONLY use information explicitly mentioned in the context above. Do NOT provide external knowledge, recipes, or information not in the context. If the context doesn't contain the answer, say "I don't have that specific information in my knowledge base."

Keep response to 2-3 sentences. Stay focused on ASAI ceramic cookware."""
        )
        
        self.conversation_history_prompt = PromptTemplate(
            input_variables=["query", "conversation_history"],
            template="""You are ASAI-Bot. The customer is asking about their previous question or conversation.

Previous conversation:
{conversation_history}

Customer Query: {query}

Look at the conversation history and accurately tell them what their last question was. Be specific and direct."""
        )

    def generate_response(self, intent: IntentType, query: str, context: ConversationContext) -> str:
        # Get relevant context from vector store
        relevant_docs = self.vector_store.similarity_search(query, k=2)
        doc_context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Format conversation history
        history_text = self._format_conversation_history(context.conversation_history)
        
        if intent == IntentType.GREETING:
            response = self.llm.invoke(self.greeting_prompt.format(
                context=doc_context, 
                conversation_history=history_text
            ))
        elif intent == IntentType.COOKING_NEED:
            # Extract dish from query
            dish = self._extract_dish(query)
            response = self.llm.invoke(self.cooking_prompt.format(
                query=query, 
                context=doc_context, 
                dish=dish,
                conversation_history=history_text
            ))
        elif intent == IntentType.PRODUCT_QUESTION:
            response = self.llm.invoke(self.product_prompt.format(
                query=query, 
                context=doc_context,
                conversation_history=history_text
            ))
        elif intent == IntentType.SIZE_INQUIRY:
            response = self.llm.invoke(self.size_prompt.format(
                context=doc_context,
                conversation_history=history_text
            ))
        elif intent == IntentType.CONVERSATION_HISTORY:
            # Handle conversation history queries
            response = self.llm.invoke(self.conversation_history_prompt.format(
                query=query, 
                conversation_history=history_text
            ))
        elif intent == IntentType.AFFIRMATIVE_RESPONSE:
            # Handle "yes", "tell me more" responses with context
            response = self.llm.invoke(self.affirmative_response_prompt.format(
                context=doc_context, 
                conversation_history=history_text
            ))
        else:
            response = self.llm.invoke(self.general_prompt.format(
                query=query, 
                context=doc_context,
                conversation_history=history_text
            ))
        
        return response.content if hasattr(response, 'content') else str(response)

    def generate_response_stream(self, intent: IntentType, query: str, context: ConversationContext):
        """Generate streaming response using LangChain's streaming capabilities"""
        # Get relevant context from vector store
        relevant_docs = self.vector_store.similarity_search(query, k=2)
        doc_context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Format conversation history
        history_text = self._format_conversation_history(context.conversation_history)
        
        # Select appropriate prompt based on intent
        if intent == IntentType.GREETING:
            prompt = self.greeting_prompt.format(
                context=doc_context,
                conversation_history=history_text
            )
        elif intent == IntentType.COOKING_NEED:
            dish = self._extract_dish(query)
            prompt = self.cooking_prompt.format(
                query=query, 
                context=doc_context, 
                dish=dish,
                conversation_history=history_text
            )
        elif intent == IntentType.PRODUCT_QUESTION:
            prompt = self.product_prompt.format(
                query=query, 
                context=doc_context,
                conversation_history=history_text
            )
        elif intent == IntentType.SIZE_INQUIRY:
            prompt = self.size_prompt.format(
                context=doc_context,
                conversation_history=history_text
            )
        elif intent == IntentType.CONVERSATION_HISTORY:
            prompt = self.conversation_history_prompt.format(
                query=query, 
                conversation_history=history_text
            )
        elif intent == IntentType.AFFIRMATIVE_RESPONSE:
            prompt = self.affirmative_response_prompt.format(
                context=doc_context, 
                conversation_history=history_text
            )
        else:
            prompt = self.general_prompt.format(
                query=query, 
                context=doc_context,
                conversation_history=history_text
            )
        
        # Create streaming LLM
        streaming_llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0, 
            streaming=True
        )
        
        # Stream the response using LangChain's streaming
        for chunk in streaming_llm.stream(prompt):
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content
    
    def _extract_dish(self, query: str) -> str:
        # Enhanced dish extraction
        dishes = {
            "noodles": ["noodle", "noodles", "ramen", "pasta", "spaghetti"],
            "dosa": ["dosa", "dosas"],
            "pancakes": ["pancake", "pancakes"],
            "eggs": ["egg", "eggs", "omelet", "omelette"],
            "stir-fry": ["stir", "stir-fry", "stir fry"],
            "crepes": ["crepe", "crepes"]
        }
        
        query_lower = query.lower()
        for dish_name, keywords in dishes.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return dish_name
        
        return "your dish"
    
    def _format_conversation_history(self, conversation_history: List[Dict]) -> str:
        """Format conversation history for context"""
        if not conversation_history or len(conversation_history) <= 1:
            return "No previous conversation"
        
        formatted = []
        for msg in conversation_history:
            role = "You" if msg.get('role') == 'user' else "ASAI-Bot"
            content = msg.get('content', '')
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)

# --- 5. PRODUCT RECOMMENDER ---

class ProductRecommender:
    def __init__(self, llm: ChatOpenAI, product_db: Dict):
        self.llm = llm
        self.product_db = product_db
        
        # LLM-based product recommendation prompt
        self.recommendation_prompt = PromptTemplate(
            input_variables=["query", "conversation_history", "available_products"],
            template="""You are an expert ASAI cookware product recommender. Analyze the customer's query and conversation to recommend the most suitable products.

Customer Query: {query}

Recent Conversation:
{conversation_history}

Available ASAI Products:
{available_products}

Based on the customer's specific needs, cooking requirements, and context, select the 2-3 MOST RELEVANT products. Consider:
1. What they want to cook
2. Portion size needs
3. Cooking method requirements
4. Any specific preferences mentioned

Return ONLY a JSON array with the product IDs of your recommendations, ordered by relevance:
["product-id-1", "product-id-2", "product-id-3"]

Examples:
- For noodles/pasta: ["asai-ceramic-saucepan-2qt", "asai-ceramic-wok-14inch", "asai-ceramic-pan-10inch"]
- For stir-fry: ["asai-ceramic-wok-14inch", "asai-ceramic-pan-12inch", "asai-ceramic-pan-10inch"]
- For dosas: ["asai-ceramic-pan-12inch", "asai-ceramic-pan-10inch", "asai-ceramic-pan-8inch"]
- For eggs/small portions: ["asai-ceramic-pan-8inch", "asai-ceramic-pan-10inch", "asai-ceramic-saucepan-2qt"]

Be specific and practical in your recommendations."""
        )
        
    def should_recommend(self, intent: IntentType, query: str) -> bool:
        """Determine if products should be recommended"""
        recommend_intents = [
            IntentType.COOKING_NEED,
            IntentType.PRODUCT_QUESTION,
            IntentType.SIZE_INQUIRY,
            IntentType.AFFIRMATIVE_RESPONSE
        ]
        
        # Don't recommend for negative responses, greetings, or conversation history queries
        if intent in [IntentType.NEGATIVE_RESPONSE, IntentType.GREETING, IntentType.CONVERSATION_HISTORY]:
            return False
            
        return intent in recommend_intents
    
    def get_recommendations(self, intent: IntentType, query: str, context: ConversationContext) -> List[Dict]:
        """Get intelligent LLM-based product recommendations"""
        if not self.should_recommend(intent, query):
            return []
        
        try:
            products = self.product_db.get("products", [])
            if not products:
                return []
            
            # Format products for LLM analysis
            product_info = self._format_products_for_llm(products)
            
            # Format conversation history
            history_text = self._format_conversation_history(context.conversation_history)
            
            # Get LLM recommendations
            response = self.llm.invoke(
                self.recommendation_prompt.format(
                    query=query,
                    conversation_history=history_text,
                    available_products=product_info
                )
            )
            
            # Parse LLM response to get product IDs
            recommended_ids = self._parse_recommendation_response(response.content)
            
            # Convert IDs to full product objects
            recommended_products = []
            for product_id in recommended_ids:
                product = next((p for p in products if p.get('id') == product_id), None)
                if product:
                    recommended_products.append(product)
            
            # Fallback to default recommendations if LLM didn't return valid products
            if not recommended_products:
                return self._get_fallback_recommendations(products)
            
            return recommended_products[:3]  # Max 3 recommendations
            
        except Exception as e:
            print(f"Error in LLM product recommendation: {e}")
            # Fallback to simple recommendations
            return self._get_fallback_recommendations(self.product_db.get("products", []))
    
    def _format_products_for_llm(self, products: List[Dict]) -> str:
        """Format products in a concise way for LLM analysis"""
        formatted_products = []
        for product in products:
            formatted = f"""
- ID: {product.get('id', '')}
  Name: {product.get('name', '')}
  Category: {product.get('category', '')}
  Size: {product.get('size', '')}
  Price: {product.get('price', '')}
  Best For: {', '.join(product.get('best_for', []))}
  Description: {product.get('description', '')}"""
            formatted_products.append(formatted)
        
        return "\n".join(formatted_products)
    
    def _format_conversation_history(self, conversation_history: List[Dict]) -> str:
        """Format conversation history for LLM context"""
        if not conversation_history or len(conversation_history) <= 1:
            return "No previous conversation"
        
        formatted = []
        for msg in conversation_history[-4:]:  # Last 4 messages for context
            role = "Customer" if msg.get('role') == 'user' else "ASAI-Bot"
            content = msg.get('content', '')
            formatted.append(f"{role}: {content}")
        
        return "\n".join(formatted)
    
    def _parse_recommendation_response(self, response_content: str) -> List[str]:
        """Parse LLM response to extract product IDs"""
        try:
            import re
            import json
            
            # Try to find JSON array in the response
            json_match = re.search(r'\[([^\]]+)\]', response_content)
            if json_match:
                json_str = json_match.group(0)
                # Clean up the JSON string
                json_str = re.sub(r'["""]', '"', json_str)  # Normalize quotes
                product_ids = json.loads(json_str)
                return [pid.strip() for pid in product_ids if isinstance(pid, str)]
            
            # Fallback: look for product IDs in text
            product_ids = re.findall(r'asai-[a-zA-Z0-9-]+', response_content)
            return product_ids[:3]
            
        except Exception as e:
            print(f"Error parsing LLM recommendation response: {e}")
            return []
    
    def _get_fallback_recommendations(self, products: List[Dict]) -> List[Dict]:
        """Fallback recommendations if LLM fails"""
        if not products:
            return []
        
        # Return most versatile products: 10-inch pan, saucepan, 8-inch pan
        fallback_order = [
            ("10 inch", "frying_pan"),
            ("2 qt", "saucepan"),
            ("8 inch", "frying_pan")
        ]
        
        recommendations = []
        for size_criteria, category_criteria in fallback_order:
            product = next((p for p in products 
                           if size_criteria in p.get('size', '') and 
                           category_criteria in p.get('category', '')), None)
            if product:
                recommendations.append(product)
        
        return recommendations[:3]

# --- 6. CONVERSATION MANAGER ---

class ConversationManager:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Initialize vector store
        self.vector_store = self._initialize_vector_store()
        
        # Initialize components
        self.intent_classifier = IntentClassifier(self.llm)
        self.response_generator = ResponseGenerator(self.llm, self.vector_store, PRODUCT_DB)
        self.product_recommender = ProductRecommender(self.llm, PRODUCT_DB)
    
    def _initialize_vector_store(self):
        """Initialize vector store with persistence. Only rebuild if needed."""
        import shutil
        
        if not os.path.exists(PDF_PATH):
            print(f"Warning: {PDF_PATH} not found")
            return None
        
        # Check if vector store already exists
        if os.path.exists("./chroma_db_modular") and os.listdir("./chroma_db_modular"):
            print("📂 Loading existing vector store from disk...")
            try:
                embedding_function = OpenAIEmbeddings()
                vector_store = Chroma(
                    persist_directory="./chroma_db_modular",
                    embedding_function=embedding_function
                )
                print(f"✅ Loaded existing vector store with {vector_store._collection.count()} documents")
                return vector_store
            except Exception as e:
                print(f"❌ Error loading existing vector store: {e}")
                print("🔄 Creating new vector store...")
        
        # Create new vector store if doesn't exist or failed to load
        print(f"📄 Loading PDF: {PDF_PATH}")    
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} pages from PDF")
        
        # Add product documents
        product_docs = self._create_product_documents()
        all_documents = documents + product_docs
        print(f"📦 Added {len(product_docs)} product documents")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        texts = text_splitter.split_documents(all_documents)
        print(f"🔨 Created {len(texts)} text chunks for vector store")
        
        print("🧠 Creating embeddings and vector store...")
        embedding_function = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embedding_function,
            persist_directory="./chroma_db_modular"
        )
        print(f"✅ Vector store created with {len(texts)} chunks")
        print(f"💾 Vector store saved to: ./chroma_db_modular")
        
        return vector_store
    
    def _create_product_documents(self):
        """Create comprehensive product documents with all details for vector store"""
        product_documents = []
        for product in PRODUCT_DB.get("products", []):
            # Create comprehensive product content for vector search
            content = f"""
Product ID: {product.get('id', '')}
Product Name: {product.get('name', '')}
Category: {product.get('category', '')}
Size: {product.get('size', '')}
Price: {product.get('price', '')}
Description: {product.get('description', '')}
Features: {'; '.join(product.get('features', []))}
Best For: {', '.join(product.get('best_for', []))}
Product URL: {product.get('product_url', '')}

Customer Reviews:
{self._format_reviews(product.get('reviews', []))}
"""
            # Include product ID and category in metadata for easy filtering
            metadata = {
                "source": "product_db",
                "product_id": product.get('id', ''),
                "category": product.get('category', ''),
                "size": product.get('size', '')
            }
            doc = Document(page_content=content, metadata=metadata)
            product_documents.append(doc)
        return product_documents
    
    def _format_reviews(self, reviews: List[Dict]) -> str:
        """Format customer reviews for vector content"""
        if not reviews:
            return "No reviews available"
        
        formatted_reviews = []
        for review in reviews[:3]:  # Limit to top 3 reviews
            rating_stars = "⭐" * review.get('rating', 0)
            author = review.get('author', 'Anonymous')
            text = review.get('text', '')
            verified = "✓ Verified" if review.get('verified', False) else ""
            formatted_reviews.append(f"{rating_stars} {author} {verified}: \"{text}\"")
        
        return "\n".join(formatted_reviews)
    
    def process_message(self, query: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Main method to process user message"""
        try:
            # 1. Classify intent (no longer needs conversation history)
            intent = self.intent_classifier.classify(query)
            
            # 2. Create conversation context
            context = ConversationContext(
                state=ConversationState.GENERAL_CHAT,
                intent=intent,
                mentioned_dishes=[],
                mentioned_sizes=[],
                user_preferences={},
                conversation_history=conversation_history
            )
            
            # 3. Generate response
            response_text = self.response_generator.generate_response(intent, query, context)
            
            # 4. Get product recommendations
            recommended_products = self.product_recommender.get_recommendations(intent, query, context)
            
            return {
                "response": response_text,
                "products": recommended_products,
                "intent": intent.value,
                "should_recommend": len(recommended_products) > 0
            }
            
        except Exception as e:
            print(f"Error processing message: {e}")
            return {
                "response": "I'm sorry, I encountered an error. How can I help you with ASAI ceramic cookware?",
                "products": [],
                "intent": "error",
                "should_recommend": False
            }

    def process_message_stream(self, query: str, conversation_history: List[Dict]):
        """Process message and return streaming response"""
        try:
            # 1. Classify intent (no longer needs conversation history)
            intent = self.intent_classifier.classify(query)
            
            # 2. Create conversation context
            context = ConversationContext(
                state=ConversationState.GENERAL_CHAT,
                intent=intent,
                mentioned_dishes=[],
                mentioned_sizes=[],
                user_preferences={},
                conversation_history=conversation_history
            )
            
            # 3. Get product recommendations quickly (use fallback for speed)
            recommended_products = []
            if self.product_recommender.should_recommend(intent, query):
                try:
                    # Try fast recommendation first
                    recommended_products = self._get_quick_recommendations(intent, query)
                except:
                    # Fallback to empty if it fails
                    recommended_products = []
            
            # 4. Stream response immediately
            response_stream = self.response_generator.generate_response_stream(intent, query, context)
            
            return {
                "response_stream": response_stream,
                "products": recommended_products,
                "intent": intent.value,
                "should_recommend": len(recommended_products) > 0
            }
            
        except Exception as e:
            print(f"Error processing message: {e}")
            def error_stream():
                yield "I'm sorry, I encountered an error. How can I help you with ASAI ceramic cookware?"
            
            return {
                "response_stream": error_stream(),
                "products": [],
                "intent": "error",
                "should_recommend": False
            }

    def _get_quick_recommendations(self, intent: IntentType, query: str) -> List[Dict]:
        """Get quick product recommendations without LLM"""
        products = PRODUCT_DB.get("products", [])
        if not products:
            return []
        
        query_lower = query.lower()
        
        # Quick pattern matching for common requests
        if any(word in query_lower for word in ["noodle", "pasta", "ramen"]):
            return [p for p in products if p.get('id') in ["asai-ceramic-saucepan-2qt", "asai-ceramic-wok-14inch"]][:2]
        elif any(word in query_lower for word in ["stir", "fry", "wok"]):
            return [p for p in products if p.get('id') in ["asai-ceramic-wok-14inch", "asai-ceramic-pan-12inch"]][:2]
        elif any(word in query_lower for word in ["dosa", "pancake", "crepe"]):
            return [p for p in products if p.get('id') in ["asai-ceramic-pan-12inch", "asai-ceramic-pan-10inch"]][:2]
        elif any(word in query_lower for word in ["egg", "small", "single"]):
            return [p for p in products if p.get('id') in ["asai-ceramic-pan-8inch", "asai-ceramic-pan-10inch"]][:2]
        else:
            # Default recommendations
            return [p for p in products if p.get('id') in ["asai-ceramic-pan-10inch", "asai-ceramic-saucepan-2qt"]][:2]

# --- 7. FLASK APPLICATION ---

# Initialize conversation manager
conversation_manager = ConversationManager()

@app.route("/ask", methods=["POST"])
def ask_assistant():
    """True streaming API endpoint with OpenAI-like response format"""
    if not request.json:
        return Response("Invalid request: JSON body required.", status=400, mimetype='text/plain')
    
    # Extract query and conversation history
    if 'messages' in request.json:
        messages = request.json['messages']
        user_messages = [msg for msg in messages if msg.get('role') == 'user']
        if not user_messages:
            return Response("No user messages found.", status=400, mimetype='text/plain')
        query_text = user_messages[-1].get('content', '')
        conversation_history = messages
    else:
        query_text = request.json.get('query', '')
        conversation_history = [{"role": "user", "content": query_text}]
    
    if not query_text:
        return Response("Query cannot be empty.", status=400, mimetype='text/plain')

    def stream_response():
        try:
            # Get streaming result
            result = conversation_manager.process_message_stream(query_text, conversation_history)
            
            # Generate response ID
            response_id = str(uuid.uuid4())
            
            # Stream tokens as they come from OpenAI
            for token in result["response_stream"]:
                chunk_data = {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": "asai-cookware-ai",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": token},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
            
            # Send finish chunk
            final_chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "asai-cookware-ai",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            
            # Send product recommendations if available
            if result["should_recommend"] and result["products"]:
                products_chunk = {
                    "id": response_id,
                    "object": "chat.completion.products",
                    "created": int(time.time()),
                    "model": "asai-cookware-ai",
                    "products": result["products"],
                    "type": "product_recommendations"
                }
                yield f"data: {json.dumps(products_chunk)}\n\n"
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            print(f"Stream error: {e}")
            error_chunk = {
                "error": {"message": "Sorry, an error occurred. Please try again.", "type": "internal_error"}
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            yield "data: [DONE]\n\n"

    return Response(stream_response(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type'
    })

@app.route("/health", methods=["GET"])
def health_check():
    return {"status": "healthy", "products_loaded": len(PRODUCT_DB.get("products", []))}

@app.route("/products", methods=["GET"])
def get_all_products():
    return {"products": PRODUCT_DB.get("products", [])}

# --- 8. MAIN ---

if __name__ == "__main__":
    print("🤖 ASAI Cookware AI Assistant (Modular) Starting...")
    print(f"📦 Loaded {len(PRODUCT_DB.get('products', []))} products")
    print(f"🚀 Server running on http://localhost:5001")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
