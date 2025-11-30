"""
Eco-IT Hardware Recommender - FastAPI + LangChain + Ollama 
Run: python main.py
Requires: Ollama running 
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json
import re

from langchain_community.llms import Ollama

app = FastAPI(
    title="Eco-IT Recommender API",
    version="1.0.0",
    description="AI-powered hardware recommendation"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# INITIALIZE OLLAMA
print("\n" + "="*60)
print("🚀 ECO-IT RECOMMENDER API - Starting")
print("="*60)

print("Initializing Ollama...")

try:
    # Initialize Ollama LLM
    llm = Ollama(
        model="deepseek-r1:1.5b",
        base_url="http://localhost:11434",
        temperature=0.3,
        num_predict=1000
    )
    
    print("Ollama initialized")
    OLLAMA_AVAILABLE = True
    
except Exception as e:
    print(f"Ollama not available: {e}")
    print("Make sure: ollama pull model && ollama serve")
    OLLAMA_AVAILABLE = False
    llm = None

# LOAD HARDWARE DATA FROM CSV 
print("Loading hardware data from CSV...")
try:
    HARDWARE_DF = pd.read_csv('../data/boavizta-data-us-normalized.csv', encoding='utf-8')
    HARDWARE_DF = HARDWARE_DF.fillna(0)
    HARDWARE_DATA = HARDWARE_DF.to_dict('records')
    
    print(f"Loaded {len(HARDWARE_DATA)} products")
    print(f"Columns: {list(HARDWARE_DF.columns)}")
    
except FileNotFoundError:
    print("file.csv not found!")
    HARDWARE_DATA = []
    HARDWARE_DF = pd.DataFrame()

except Exception as e:
    print(f"Error: {e}")
    HARDWARE_DATA = []
    HARDWARE_DF = pd.DataFrame()

# DATA MODELS 
class UserRequest(BaseModel):
    requirements: str


# ============ REQUIREMENT EXTRACTION WITH LLM ============

def extract_requirements_with_langchain(user_input: str) -> dict:
    """
    Use LangChain + Ollama to extract and classify requirements
    LLM handles ALL classification (use_case, product_type, priority, etc.)
    Single LLM call returns everything needed
    """
    
    if OLLAMA_AVAILABLE and llm:
        try:
            print(f"🔗 LangChain calling Ollama...")
            
            prompt = f"""You are a hardware classification expert. Be VERY PRECISE.

                User request: "{user_input}"

                TASK: Extract and classify the hardware requirements.

                RULES:
                1. If user says "laptop", ALWAYS set product_type to "laptop" (NOT monitor, NOT desktop)
                2. If user says "monitor" or "screen", ALWAYS set product_type to "monitor"
                3. If user says "desktop" or "tower", ALWAYS set product_type to "desktop"
                4. If user says "server" or "rack", ALWAYS set product_type to "server"
                5. Only set product_type if explicitly mentioned or VERY clear from context
                6. If product_type is unclear, set to "unknown" and let the system expand search

                CLASSIFICATION FIELDS:

                1. use_case (REQUIRED - pick ONE):
                - "workplace" = office, company, business, work environment
                - "datacenter" = servers, infrastructure, data center, cloud
                - "home" = personal, home office, living space, family
                - "unknown" = cannot determine

                2. product_type (REQUIRED - pick ONE or "unknown"):
                - "laptop" = portable computer, notebook, ultrabook
                - "desktop" = tower, workstation, all-in-one (non-portable)
                - "monitor" = screen, display (NOT a computer)
                - "server" = server, rack server, blade
                - "tablet" = tablet, iPad
                - "phone" = phone, smartphone
                - "watch" = smartwatch, wearable
                - "unknown" = cannot determine from context

                3. priority (pick ONE):
                - "energy_efficiency" = green, eco, low power, efficient, carbon, sustainable, battery life
                - "performance" = fast, powerful, gaming, video editing, rendering, AI, high-performance, compute-intensive
                - "cost" = cheap, budget, affordable, low price, economical
                - "balanced" = no clear priority, mixed requirements, or normal use

                4. memory (estimate RAM in GB):
                - For laptop: typical 8-16, gaming/video: 32-64
                - For desktop: typical 8-16, workstation: 32-64+
                - For server: 64-512+

                5. number_cpu (estimate cores/threads):
                - For laptop: 4-8 cores
                - For desktop: 6-16 cores
                - For server: 16-64+ cores

                6. budget (pick ONE):
                - "low" = budget, limited budget, cheap
                - "medium" = normal budget, standard
                - "high" = premium, expensive, "high budget", no budget constraints

                RESPOND ONLY with valid JSON (NO other text):
                {{
                    "use_case": "workplace" | "datacenter" | "home" | "unknown",
                    "product_type": "laptop" | "desktop" | "monitor" | "server" | "tablet" | "phone" | "watch" | "unknown",
                    "priority": "energy_efficiency" | "performance" | "cost" | "balanced",
                    "memory": <integer GB>,
                    "number_cpu": <integer cores>,
                    "budget": "low" | "medium" | "high",
                    "confidence": <float 0.0-1.0>,
                    "reasoning": "<explanation of classification>"
                }}

                CRITICAL TEST CASES:
                - "laptop" → product_type: "laptop" (NEVER "monitor" or "desktop")
                - "monitor" → product_type: "monitor" (NEVER "laptop")
                - "desktop computer" → product_type: "desktop" (NEVER "monitor")
                - "server for datacenter" → product_type: "server"

                NOW CLASSIFY THIS REQUEST:
                "{user_input}"

                Remember: Be PRECISE. If unsure about product_type, set to "unknown" rather than guess wrong."""
            
            result = llm.invoke(prompt)
            print(f"📝 LLM response: {result}")
            
            # Parse JSON from response
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            
            if json_match:
                requirements = json.loads(json_match.group(0))
                
                # Validate required fields
                required_fields = ['use_case', 'product_type', 'priority', 'memory', 'number_cpu']
                if all(field in requirements for field in required_fields):
                    print(f"✅ Extracted: {requirements['use_case']} → {requirements['product_type']}")
                    print(f"   Priority: {requirements['priority']} | Memory: {requirements.get('memory')}GB | CPUs: {requirements.get('number_cpu')}")
                    print(f"   Budget: {requirements.get('budget', 'medium')} | Confidence: {requirements.get('confidence', 0):.0%}")
                    print(f"   Reasoning: {requirements.get('reasoning', 'N/A')}")
                    return requirements
                else:
                    print(f"⚠️  LLM response missing required fields, using fallback")
                    print(f"   Got: {list(requirements.keys())}")
        
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing error: {e}")
            print(f"   Response was: {result}")
        except Exception as e:
            print(f"⚠️  LangChain error: {e}")
    
    # ============ FALLBACK: Safe defaults ============
    print(f"⚠️  Using fallback parsing")
    
    # Try basic keyword detection for fallback
    user_lower = user_input.lower()
    
    product_type = "unknown"
    if "laptop" in user_lower or "notebook" in user_lower:
        product_type = "laptop"
    elif "desktop" in user_lower or "tower" in user_lower:
        product_type = "desktop"
    elif "monitor" in user_lower or "screen" in user_lower or "display" in user_lower:
        product_type = "monitor"
    elif "server" in user_lower:
        product_type = "server"
    elif "tablet" in user_lower:
        product_type = "tablet"
    
    use_case = "unknown"
    if "company" in user_lower or "office" in user_lower or "workplace" in user_lower:
        use_case = "workplace"
    elif "datacenter" in user_lower or "data center" in user_lower:
        use_case = "datacenter"
    elif "home" in user_lower:
        use_case = "home"
    
    priority = "balanced"
    if "performance" in user_lower or "fast" in user_lower or "powerful" in user_lower:
        priority = "performance"
    elif "energy" in user_lower or "green" in user_lower or "efficient" in user_lower:
        priority = "energy_efficiency"
    elif "cheap" in user_lower or "budget" in user_lower:
        priority = "cost"
    
    budget = "medium"
    if "high budget" in user_lower or "expensive" in user_lower or "premium" in user_lower:
        budget = "high"
    elif "cheap" in user_lower or "budget" in user_lower:
        budget = "low"
    
    requirements = {
        "use_case": use_case,
        "product_type": product_type,
        "priority": priority,
        "budget": budget,
        "memory": 16,
        "number_cpu": 4,
        "confidence": 0.4,
        "reasoning": "Fallback parsing from keywords"
    }
    
    print(f"   Fallback result: {product_type} in {use_case} with {priority} priority")
    
    return requirements



# ============ CATEGORY & SUBCATEGORY EXTRACTION ============

# def get_category_from_requirements(requirements: dict) -> tuple:
#     """
#     Get category and subcategory from LLM-extracted requirements
#     No need to search/infer - LLM already did it!
#     """
    
#     category = requirements.get('use_case', 'unknown')
#     subcategory = requirements.get('product_type', 'unknown')
#     confidence = requirements.get('confidence', 0)
    
#     # Map to proper capitalization for database matching
#     category_map = {
#         'workplace': 'Workplace',
#         'home': 'Home',
#         'datacenter': 'Datacenter',
#         'unknown': 'Unknown'
#     }
    
#     category = category_map.get(category.lower(), 'Unknown')
    
#     print(f"🏷️  Category: {category} | Subcategory: {subcategory} | Confidence: {confidence:.0%}")
    
#     return category, subcategory

def get_category_from_requirements(requirements: dict) -> tuple:
    """
    Get category and subcategory from LLM-extracted requirements
    Maps to CSV columns: category and subcategory
    """
    
    use_case = requirements.get('use_case', 'unknown').lower()
    product_type = requirements.get('product_type', 'unknown').lower()
    confidence = requirements.get('confidence', 0)
    
    # Map use_case to category
    category_map = {
        'workplace': 'Workplace',
        'home': 'Home',
        'datacenter': 'Datacenter',
        'unknown': 'Unknown'
    }
    
    category = category_map.get(use_case, 'Unknown')
    
    # Map product_type to subcategory (capitalize first letter)
    subcategory_map = {
        'laptop': 'Laptop',
        'desktop': 'Desktop',
        'monitor': 'Monitor',
        'server': 'Server',
        'tablet': 'Tablet',
        'phone': 'Phone',
        'watch': 'Watch',
        'unknown': 'unknown'
    }
    
    subcategory = subcategory_map.get(product_type, product_type.capitalize() if product_type != 'unknown' else 'unknown')
    
    print(f"🏷️  LLM: use_case={use_case} → category={category}")
    print(f"🏷️  LLM: product_type={product_type} → subcategory={subcategory}")
    print(f"🏷️  Confidence: {confidence:.0%}")
    
    return category, subcategory


# ============ FILTER RELEVANT PRODUCTS ============

def filter_relevant_products(requirements: dict) -> list:
    """
    Filter products based on LLM-extracted requirements
    Uses BOTH category and subcategory columns from CSV
    """
    
    category, subcategory = get_category_from_requirements(requirements)
    
    if category == 'Unknown' or subcategory == 'unknown':
        print(f"⚠️  Could not classify - returning all products")
        return HARDWARE_DATA
    
    filtered = []
    
    for hw in HARDWARE_DATA:
        hw_category = str(hw.get('category', '')).lower()
        hw_subcategory = str(hw.get('subcategory', '')).lower()
        
        # Match category (e.g., "Workplace")
        if category.lower() not in hw_category:
            continue
        
        # Match subcategory (e.g., "Laptop")
        if subcategory.lower() not in hw_subcategory:
            continue
        
        filtered.append(hw)
    
    print(f"✅ Filtered from {len(HARDWARE_DATA)} to {len(filtered)} {subcategory}s in {category}")
    
    if not filtered:
        print(f"⚠️  No {subcategory}s found in {category}")
        
        # Debug: show what subcategories exist in this category
        available_subcats = set()
        for hw in HARDWARE_DATA:
            if category.lower() in str(hw.get('category', '')).lower():
                available_subcats.add(str(hw.get('subcategory', '')))
        
        print(f"   Available subcategories in {category}: {sorted(available_subcats)}")
        
        # Fallback: return all in category
        print(f"   Expanding to all {category} products...")
        for hw in HARDWARE_DATA:
            hw_category = str(hw.get('category', '')).lower()
            if category.lower() in hw_category:
                filtered.append(hw)
        
        print(f"✅ Expanded to {len(filtered)} products in {category}")
    
    return filtered



# ============ ENVIRONMENTAL COST SCORING ============

def calculate_environmental_score(hw: dict) -> float:
    """
    Calculate environmental cost score (0-100)
    Lower environmental impact = higher score
    
    Factors:
    - GWP (manufacturing + use): 40%
    - Yearly energy consumption: 30%
    - Manufacturing ratio: 20%
    - Lifetime efficiency: 10%
    """
    
    try:
        gwp_total = float(hw.get('gwp_total', 500)) or 500
        yearly_tec = float(hw.get('yearly_tec', 100)) or 100
        lifetime = float(hw.get('lifetime', 5)) or 5
        gwp_mfg_ratio = float(hw.get('gwp_manufacturing_ratio', 0.5)) or 0.5
    except (ValueError, TypeError):
        return 0
    
    # 1. GWP Score (40%) - Lower is better
    # Normalize: 0-2000 kg CO2eq → 100-0 score
    gwp_score = max(0, min(100, 100 - (gwp_total / 20)))
    
    # 2. Yearly Energy Score (30%) - Lower is better
    # Normalize: 0-500 kWh → 100-0 score
    tec_score = max(0, min(100, 100 - (yearly_tec / 5)))
    
    # 3. Manufacturing Ratio Score (20%) - Lower mfg impact is better
    # If most impact is in use phase, better for environment (can reduce over time)
    # If most impact is in manufacturing, worse (locked in)
    use_phase_ratio = 1 - gwp_mfg_ratio
    mfg_score = use_phase_ratio * 100
    
    # 4. Lifetime Efficiency Score (10%) - Longer lifetime is better
    # Normalize: 3-10 years → 0-100 score
    lifetime_score = max(0, min(100, (lifetime - 3) / 7 * 100))
    
    # Weighted average
    env_score = (
        gwp_score * 0.4 +
        tec_score * 0.3 +
        mfg_score * 0.2 +
        lifetime_score * 0.1
    )
    
    return min(100, max(0, env_score))


# ============ PERFORMANCE SCORE ============

def calculate_performance_score(hw: dict, requirements: dict) -> float:
    """
    Calculate performance score (0-100)
    Based on how well specs match user requirements
    """
    
    try:
        req_memory = float(requirements.get('memory', 16)) or 16
        req_cpu = float(requirements.get('number_cpu', 4)) or 4
        
        hw_memory = float(hw.get('memory', 16)) or 16
        hw_cpu = float(hw.get('number_cpu', 4)) or 4
    except (ValueError, TypeError):
        return 50
    
    # Memory matching (60%)
    memory_ratio = hw_memory / req_memory if req_memory > 0 else 1
    # Penalize if too low, reward if meets/exceeds
    memory_score = min(100, memory_ratio * 100)
    if memory_ratio < 0.8:
        memory_score *= 0.5  # Penalty for insufficient RAM
    
    # CPU matching (40%)
    cpu_ratio = hw_cpu / req_cpu if req_cpu > 0 else 1
    cpu_score = min(100, cpu_ratio * 100)
    if cpu_ratio < 0.8:
        cpu_score *= 0.5  # Penalty for insufficient CPUs
    
    perf_score = (memory_score * 0.4 + cpu_score * 0.6)
    
    return min(100, max(0, perf_score))


# ============ MAIN SCORING FUNCTION ============

def score_hardware(hw: dict, requirements: dict) -> float:
    """
    Score hardware based on user priorities
    
    Weights:
    - Energy efficiency priority: 70% environmental, 30% performance
    - Performance priority: 30% environmental, 70% performance
    - Cost priority: 50% environmental, 50% performance
    - Balanced: 50% environmental, 50% performance
    """
    
    priority = requirements.get('priority', 'balanced')
    
    env_score = calculate_environmental_score(hw)
    perf_score = calculate_performance_score(hw, requirements)
    
    if priority == 'energy_efficiency':
        final_score = env_score * 0.8 + perf_score * 0.2
    elif priority == 'performance':
        final_score = env_score * 0.2 + perf_score * 0.8
    elif priority == 'cost':
        final_score = env_score * 0.5 + perf_score * 0.5
    else:  # balanced
        final_score = env_score * 0.5 + perf_score * 0.5
    
    return min(100, max(0, final_score))


# ============ FIND RECOMMENDATIONS (OPTIMIZED) ============

def find_recommendations(requirements: dict, top_n: int = 5) -> list:
    """
    Optimized recommendation engine:
    1. Filter to relevant products (category + subcategory)
    2. Score only relevant products
    3. Return top N ranked by score
    """
    
    print(f"\n🔍 Step 1: Filtering relevant products...")
    relevant_products = filter_relevant_products(requirements)
    
    if not relevant_products:
        print(f"⚠️  No relevant products found!")
        return []

    print(f"🔍 Step 2: Scoring {len(relevant_products)} products...")
    candidates = []
    
    for hw in relevant_products:
        score = score_hardware(hw, requirements)
        hw_with_score = hw.copy()
        hw_with_score['score'] = score
        candidates.append(hw_with_score)
    
    print(f"🔍 Step 3: Sorting and selecting top {top_n}...")
    candidates.sort(key=lambda x: x['score'], reverse=True)
    
    top_recommendations = candidates[:top_n]
    
    for i, hw in enumerate(top_recommendations, 1):
        print(f"  #{i}: {hw.get('manufacturer')} {hw.get('name')} - Score: {hw['score']:.1f}")
    
    return top_recommendations


# ============ GENERATE REASONING WITH LANGCHAIN ============

def generate_reasoning_with_langchain(hw: dict, requirements: dict, user_input: str) -> str:
    """
    Generate reasoning highlighting environmental benefits
    Uses LLM to create natural language explanations
    """
    
    if OLLAMA_AVAILABLE and llm:
        try:
            priority = requirements.get('priority', 'balanced')
            env_score = calculate_environmental_score(hw)
            
            prompt = f"""You are a sustainable IT expert.
                    User asked: {user_input[:80]}
                    Recommended: {hw.get('manufacturer')} {hw.get('name')}
                    Priority: {priority}

                    Environmental metrics:
                    - GWP (lifecycle): {hw.get('gwp_total', 0):.0f} kgCO₂eq
                    - Yearly energy: {hw.get('yearly_tec', 0):.0f} kWh
                    - Manufacturing ratio: {hw.get('gwp_manufacturing_ratio', 0)*100:.0f}%
                    - Lifetime: {hw.get('lifetime', 5)} years
                    - Environmental score: {env_score:.0f}/100

                    Write 2 sentences explaining why this is a good recommendation.
                    Always mention environmental benefits. Be technical and concise."""
            
            result = llm.invoke(prompt)
            return result.strip()
        
        except Exception as e:
            print(f"⚠️  Reasoning error: {e}")
    
    # Fallback reasoning - emphasize environmental
    priority = requirements.get('priority', 'balanced')
    env_score = calculate_environmental_score(hw)
    gwp = hw.get('gwp_total', 0)
    energy = hw.get('yearly_tec', 0)
    
    if env_score >= 70:
        env_rating = "excellent environmental performance"
    elif env_score >= 50:
        env_rating = "good environmental profile"
    else:
        env_rating = "moderate environmental footprint"
    
    if priority == 'energy_efficiency':
        return f"Outstanding choice with {env_rating}. Only {energy:.0f} kWh/year consumption and {gwp:.0f} kgCO₂eq lifecycle impact."
    elif priority == 'performance':
        return f"Strong specs meet performance needs while maintaining {env_rating}. {gwp:.0f} kgCO₂eq environmental cost."
    else:
        return f"Well-balanced option with {env_rating}. {gwp:.0f} kgCO₂eq lifetime emissions and {energy:.0f} kWh/year usage."


# ============ GENERATE COMPARISON ============

def generate_comparison(recommendations: list) -> dict:
    """
    Generate comparison metrics across recommendations
    """
    if not recommendations:
        return {}
    
    try:
        return {
            "gwp_comparison": [
                {
                    "name": f"{h.get('manufacturer', 'Unknown')} {h.get('name', 'Unknown')}",
                    "gwp": float(h.get('gwp_total', 0)) if h.get('gwp_total') else 0
                }
                for h in recommendations
            ],
            "energy_comparison": [
                {
                    "name": f"{h.get('manufacturer', 'Unknown')} {h.get('name', 'Unknown')}",
                    "yearly_tec": float(h.get('yearly_tec', 0)) if h.get('yearly_tec') else 0
                }
                for h in recommendations
            ],
            "lifecycle_cost": [
                {
                    "name": f"{h.get('manufacturer', 'Unknown')} {h.get('name', 'Unknown')}",
                    "total_emissions": round(
                        (float(h.get('gwp_total', 0)) if h.get('gwp_total') else 0) + 
                        ((float(h.get('yearly_tec', 0)) if h.get('yearly_tec') else 0) * 
                         (float(h.get('lifetime', 5)) if h.get('lifetime') else 5) / 1000),
                        2
                    )
                }
                for h in recommendations
            ],
            "average_gwp": round(
                np.mean([float(h.get('gwp_total', 0)) if h.get('gwp_total') else 0 for h in recommendations]),
                2
            ),
            "average_yearly_tec": round(
                np.mean([float(h.get('yearly_tec', 0)) if h.get('yearly_tec') else 0 for h in recommendations]),
                2
            ),
        }
    except Exception as e:
        print(f"⚠️  Comparison error: {e}")
        return {}










# API ENDPOINTS

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "hardware_count": len(HARDWARE_DATA),
        "ollama_available": OLLAMA_AVAILABLE,
        "backend": "LangChain + Ollama"
    }

@app.post("/recommend")
def recommend_hardware(request: UserRequest) -> dict:
    """
    Main endpoint: Accept user requirements, return recommendations
    Uses LangChain for intelligent requirement extraction and reasoning
    """
    try:
        if not request.requirements.strip():
            raise HTTPException(status_code=400, detail="Requirements cannot be empty")
        
        if not HARDWARE_DATA:
            raise HTTPException(status_code=500, detail="Hardware data not loaded")
        
        # Step 1: Extract requirements with LangChain
        print(f"\n📝 Processing: {request.requirements[:50]}...")
        requirements = extract_requirements_with_langchain(request.requirements)
        print(f"✅ Extracted: {requirements}")
        
        # Step 2: Find matching hardware
        recommendations = find_recommendations(requirements, top_n=3)
        
        if not recommendations:
            raise HTTPException(status_code=404, detail="No hardware found")
        
        # Step 3: Build response
        result = {
            "user_input": request.requirements,
            "extracted_requirements": requirements,
            "recommendations": [],
            "comparison": {}
        }
        
        # Step 4: Generate reasoning for each recommendation with LangChain
        for hw in recommendations:
            reasoning = generate_reasoning_with_langchain(hw, requirements, request.requirements)
            
            result["recommendations"].append({
                "name": hw.get('name', 'Unknown'),
                "manufacturer": hw.get('manufacturer', 'Unknown'),
                "category": hw.get('category', 'Unknown'),
                "gwp_total": float(hw.get('gwp_total', 0)) if hw.get('gwp_total') else 0,
                "gwp_manufacturing_ratio": float(hw.get('gwp_manufacturing_ratio', 0)) if hw.get('gwp_manufacturing_ratio') else 0,
                "gwp_use_ratio": float(hw.get('gwp_use_ratio', 0)) if hw.get('gwp_use_ratio') else 0,
                "yearly_tec": float(hw.get('yearly_tec', 0)) if hw.get('yearly_tec') else 0,
                "lifetime": int(hw.get('lifetime', 5)) if hw.get('lifetime') else 5,
                "use_location": str(hw.get('use_location', 'WW')),
                "memory": hw.get('memory', 'N/A'),
                "number_cpu": hw.get('number_cpu', 'N/A'),
                "height": hw.get('height', 'N/A'),
                "score": round(hw['score'], 2),
                "reasoning": reasoning
            })
        
        # Step 5: Generate comparison
        result["comparison"] = generate_comparison(recommendations)
        
        print(f"✅ Recommendations ready\n")
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hardware/stats")
def hardware_stats():
    """Get statistics about loaded hardware"""

    if not HARDWARE_DATA:
        return {"error": "No data loaded"}
    
    try:
        gwp_values = [float(h.get('gwp_total', 0)) for h in HARDWARE_DATA if h.get('gwp_total')]
        tec_values = [float(h.get('yearly_tec', 0)) for h in HARDWARE_DATA if h.get('yearly_tec')]
        
        return {
            "total_products": len(HARDWARE_DATA),
            "manufacturers": sorted(list(set([h.get('manufacturer', 'Unknown') for h in HARDWARE_DATA]))),
            "categories": sorted(list(set([h.get('category', 'Unknown') for h in HARDWARE_DATA]))),
            "regions": sorted(list(set([h.get('use_location', 'WW') for h in HARDWARE_DATA]))),
            "avg_gwp": round(np.mean(gwp_values), 2) if gwp_values else 0,
            "avg_yearly_tec": round(np.mean(tec_values), 2) if tec_values else 0,
        }
    except Exception as e:
        return {"error": str(e)}

# STARTUP 
@app.on_event("startup")
async def startup_event():
    print(f"Loaded {len(HARDWARE_DATA)} products")
    
    if OLLAMA_AVAILABLE:
        print("Ollama ready")
    else:
        print("Using fallback mode (no LLM)")
    
    print(f"\nAPI running on http://0.0.0.0:8000")
    print(f"Docs: http://localhost:8000/docs")
    print("="*60 + "\n")

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
