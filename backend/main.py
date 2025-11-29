"""
Eco-IT Hardware Recommender - FastAPI + LangChain + Ollama (Simplified)
Run: python main.py
Requires: Ollama running (ollama serve in another terminal)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import json
import re

# ✅ Simplified LangChain imports
from langchain_community.llms import Ollama

app = FastAPI(
    title="Eco-IT Recommender API",
    version="1.0.0",
    description="AI-powered hardware recommendation with LangChain + Ollama"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ INITIALIZE LANGCHAIN + OLLAMA ============
print("\n" + "="*60)
print("🚀 ECO-IT RECOMMENDER API - Starting")
print("="*60)

print("🤖 Initializing LangChain + Ollama...")

try:
    # Initialize Ollama LLM - Direct usage (no chains)
    llm = Ollama(
        model="gemma3:1b",
        base_url="http://localhost:11434",
        temperature=0.3,
        num_predict=256
    )
    
    print("✅ LangChain + Ollama initialized")
    OLLAMA_AVAILABLE = True
    
except Exception as e:
    print(f"⚠️  Ollama not available: {e}")
    print("   Make sure: ollama pull mistral && ollama serve")
    OLLAMA_AVAILABLE = False
    llm = None

# ============ LOAD HARDWARE DATA FROM CSV ============
print("📂 Loading hardware data from CSV...")
try:
    HARDWARE_DF = pd.read_csv('../data/boavizta-data-us.csv', encoding='utf-8')
    HARDWARE_DF = HARDWARE_DF.fillna(0)
    HARDWARE_DATA = HARDWARE_DF.to_dict('records')
    
    print(f"✅ Loaded {len(HARDWARE_DATA)} products")
    print(f"📋 Columns: {list(HARDWARE_DF.columns)}")
    
except FileNotFoundError:
    print(" file.csv not found!")
    HARDWARE_DATA = []
    HARDWARE_DF = pd.DataFrame()

except Exception as e:
    print(f"Error: {e}")
    HARDWARE_DATA = []
    HARDWARE_DF = pd.DataFrame()

# ============ DATA MODELS ============
class UserRequest(BaseModel):
    requirements: str

# ============ REQUIREMENT EXTRACTION WITH LANGCHAIN ============

def extract_requirements_with_langchain(user_input: str) -> dict:
    """
    Use LangChain + Ollama to extract structured requirements
    Falls back to heuristic parsing if LLM unavailable
    """
    
    if OLLAMA_AVAILABLE and llm:
        try:
            print(f"LangChain calling Ollama...")
            
            # Create prompt
            prompt = f"""You are a hardware requirements analyzer.
            User request: "{user_input}"

            Extract the following in JSON format only (no other text):
            - use_case: (workplace, datacenter, or unknown)
            - priority: (energy_efficiency, performance, cost, balanced)
            - budget: (low, medium, high)
            - hard_drive: (the hard drive of the device if any)
            - memory: (RAM in GB)
            - number_cpu: number of CPUs
            - height: the height of the device in a datacenter rack, in U
            - region: (US, EU, CN, WW, or not specified)

            ONLY respond with valid JSON. Example:
            {{"use_case": "datacenter", "priority": "energy_efficiency", "budget": "high", "region": "EU"}}"""
            
            # Call LLM directly
            result = llm.invoke(prompt)
            
            print(f"📝 LLM response: {result[:100]}...")
            
            # Parse JSON from response
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            
            if json_match:
                parsed = json.loads(json_match.group(0))
                print(f"✅ Extracted: {parsed}")
                return parsed
        
        except Exception as e:
            print(f"⚠️  LangChain error: {e}, using fallback")
    
    # ============ FALLBACK: Heuristic parsing ============
    print(f"⚠️  Using fallback parsing")
    
    requirements = {
        "use_case": "unknown",
        "priority": "balanced",
        "budget": "medium",
        "region": "WW"
    }
    
    user_lower = user_input.lower()
    
    # Detect use case
    if any(word in user_lower for word in ['server', 'datacenter', 'data center', 'cluster', 'rack', 'switch']):
        requirements['use_case'] = 'datacenter'
    elif any(word in user_lower for word in ['laptop', 'desktop', 'workstation', 'computer', 'office', 'workplace']):
        requirements['use_case'] = 'workplace'
    
    # Detect priority
    if any(word in user_lower for word in ['energy', 'efficient', 'eco', 'carbon', 'green', 'sustainable', 'footprint']):
        requirements['priority'] = 'energy_efficiency'
    elif any(word in user_lower for word in ['performance', 'fast', 'speed', 'power', 'ai', 'machine learning']):
        requirements['priority'] = 'performance'
    elif any(word in user_lower for word in ['cheap', 'budget', 'cost', 'low price', 'limited']):
        requirements['priority'] = 'cost'
    
    # Detect budget
    if any(word in user_lower for word in ['expensive', 'high budget', 'no budget', 'unlimited', 'premium']):
        requirements['budget'] = 'high'
    elif any(word in user_lower for word in ['cheap', 'budget', 'limited', 'low cost']):
        requirements['budget'] = 'low'
    
    # Detect region
    if 'us ' in user_lower or 'usa' in user_lower:
        requirements['region'] = 'US'
    elif 'eu' in user_lower or 'europe' in user_lower:
        requirements['region'] = 'EU'
    elif 'china' in user_lower or 'cn' in user_lower:
        requirements['region'] = 'CN'
    
    return requirements

# ============ REASONING GENERATION WITH LANGCHAIN ============

def generate_reasoning_with_langchain(hw: dict, requirements: dict, user_input: str) -> str:
    """
    Use LangChain to generate natural language reasoning
    """
    
    if OLLAMA_AVAILABLE and llm:
        try:
            priority = requirements.get('priority', 'balanced')
            
            # Determine focus area
            if priority == 'energy_efficiency':
                focus_area = "energy efficiency and low carbon footprint"
            elif priority == 'performance':
                focus_area = "performance and computing power"
            elif priority == 'cost':
                focus_area = "cost-effectiveness"
            else:
                focus_area = "overall balance"
            
            prompt = f"""You are a sustainable IT expert.
            User asked: {user_input[:100]}
            Recommended: {hw.get('manufacturer', 'Unknown')} {hw.get('name', 'Unknown')}
            Priority: {priority}
            GWP: {hw.get('gwp_total', 0)} kgCO2eq
            Yearly Energy: {hw.get('yearly_tec', 0)} kWh

            Write a 1-2 sentence explanation why this is a good recommendation.
            Be concise and technical. Focus on: {focus_area}"""
            
            result = llm.invoke(prompt)
            return result.strip()
        
        except Exception as e:
            print(f"Reasoning generation error: {e}")
    
    # Fallback reasoning
    priority = requirements.get('priority', 'balanced')
    score = hw.get('score', 0)
    
    if priority == 'energy_efficiency':
        return f"Excellent choice for energy efficiency (Score: {score:.0f}/100). Lower energy consumption and manufacturing impact."
    elif priority == 'performance':
        return f"Strong performance option (Score: {score:.0f}/100). More CPUs and memory for demanding workloads."
    elif priority == 'cost':
        return f"Cost-effective solution (Score: {score:.0f}/100). Compact form factor and reasonable specifications."
    else:
        return f"Well-balanced across all metrics (Score: {score:.0f}/100)."

# ============ SCORING & RANKING ============

def score_hardware(hw: dict, requirements: dict) -> float:
    """
    Score hardware 0-100 based on user requirements
    """
    score = 50
    
    # Category match (40 points)
    use_case = requirements.get('use_case', 'unknown')
    hw_category = str(hw.get('category', '')).lower()
    
    if use_case != 'unknown':
        if (use_case == 'datacenter' and 'datacenter' in hw_category) or \
           (use_case == 'workplace' and 'workplace' in hw_category):
            score += 40
    
    # Priority-based scoring
    priority = requirements.get('priority', 'balanced')
    
    try:
        yearly_tec = float(hw.get('yearly_tec', 5000)) or 5000
        gwp_total = float(hw.get('gwp_total', 500)) or 500
        memory = float(hw.get('memory', 0)) or 0
        number_cpu = float(hw.get('number_cpu', 1)) or 1
        height = float(hw.get('height', 2)) or 2
    except (ValueError, TypeError):
        yearly_tec = 5000
        gwp_total = 500
        memory = 0
        number_cpu = 1
        height = 2
    
    if priority == 'energy_efficiency':
        tec_score = max(0, 100 - (yearly_tec / 100))
        score += tec_score * 0.3
        
        gwp_score = max(0, 100 - (gwp_total / 50))
        score += gwp_score * 0.2
    
    elif priority == 'performance':
        cpu_score = min(100, (number_cpu / 32) * 100)
        score += cpu_score * 0.3
        
        mem_score = min(100, (memory / 512) * 100)
        score += mem_score * 0.2
    
    elif priority == 'cost':
        size_score = 100 - min(100, (height / 10) * 100)
        score += size_score * 0.25
    
    else:  # balanced
        tec_score = max(0, 100 - (yearly_tec / 100))
        gwp_score = max(0, 100 - (gwp_total / 50))
        score += (tec_score + gwp_score) / 2 * 0.25
    
    # Region match (10 points)
    region = requirements.get('region', 'WW')
    hw_region = str(hw.get('use_location', 'WW')).upper()
    
    if hw_region == region or hw_region == 'WW':
        score += 10
    
    return min(100, max(0, score))

def find_recommendations(requirements: dict, top_n: int = 5) -> list:
    """
    Find top N hardware recommendations
    """
    candidates = []
    
    for hw in HARDWARE_DATA:
        if not hw.get('name') or not hw.get('manufacturer'):
            continue
        
        score = score_hardware(hw, requirements)
        hw_with_score = hw.copy()
        hw_with_score['score'] = score
        candidates.append(hw_with_score)
    
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:top_n]

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

# ============ API ENDPOINTS ============

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

# ============ STARTUP ============
@app.on_event("startup")
async def startup_event():
    print(f"📊 Loaded {len(HARDWARE_DATA)} products")
    
    if OLLAMA_AVAILABLE:
        print("✅ LangChain + Ollama ready")
    else:
        print("⚠️  Using fallback mode (no LLM)")
    
    print(f"\n📡 API running on http://0.0.0.0:8000")
    print(f"📚 Docs: http://localhost:8000/docs")
    print("="*60 + "\n")

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
