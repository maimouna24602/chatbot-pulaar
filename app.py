"""
app.py — Chatbot Pulaar (Streamlit)
Basé sur Notebook_NLP_pular.ipynb
Modèle : GPT-2 fine-tuné sur dialogues + QA Pulaar
"""

import random
import torch
import streamlit as st
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# ══════════════════════════════════════════════════════
#  CONFIG PAGE
# ══════════════════════════════════════════════════════
st.set_page_config(page_title="Chatbot Pulaar 🌿", page_icon="🌿", layout="centered")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #fdf6e3 0%, #fef9ee 100%); }
.msg-user {
    background: #e8f5e9; border-radius: 18px 18px 4px 18px;
    padding: 10px 16px; margin: 8px 0 8px 50px;
    border-left: 4px solid #4caf50; font-size: 1.05em;
}
.msg-bot {
    background: #fff8e1; border-radius: 18px 18px 18px 4px;
    padding: 10px 16px; margin: 8px 50px 8px 0;
    border-left: 4px solid #ff9800; font-size: 1.05em;
}
.badge {
    font-size: 0.7em; background: #795548; color: white;
    border-radius: 8px; padding: 1px 7px; margin-left: 6px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  CHARGEMENT MODÈLE
# ══════════════════════════════════════════════════════
FINETUNED_PATH  = "./pulaar_model_finetuned"
PRETRAINED_PATH = "./pulaar_model_pretrained"

@st.cache_resource(show_spinner="Chargement du modèle Pulaar…")
def load_model():
    pulaar_chars = ['\u0253', '\u0257', '\u01b4', '\u014b']  # ɓ ɗ ƴ ŋ

    for path, label in [(FINETUNED_PATH, "fine-tuné"), (PRETRAINED_PATH, "pré-entraîné")]:
        if Path(path).exists():
            tok   = AutoTokenizer.from_pretrained(path)
            tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(path)
            model.eval()
            return tok, model, label

    # Fallback : GPT-2 brut + tokens Pulaar
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    new_tokens = [ch for ch in pulaar_chars if ch not in tok.vocab]
    tok.add_tokens(new_tokens)
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.resize_token_embeddings(len(tok))
    model.eval()
    return tok, model, "GPT-2 brut (démo)"


# ══════════════════════════════════════════════════════
#  GENERATE — identique au notebook
# ══════════════════════════════════════════════════════
def generate(prompt, model, tok, max_new=60, temperature=0.7, top_p=0.9):
    device = next(model.parameters()).device
    ids    = tok.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=max_new,
            temperature=temperature, top_p=top_p,
            do_sample=True, pad_token_id=tok.eos_token_id,
        )
    full = tok.decode(out[0], skip_special_tokens=True)
    if "R:" in full:
        answer = full.split("R:", 1)[-1].strip()
        if "\nQ:" in answer:
            answer = answer.split("\nQ:")[0].strip()
        return answer
    return full.strip()


# ══════════════════════════════════════════════════════
#  RÉPONSES DE SECOURS (dataset du notebook)
# ══════════════════════════════════════════════════════
FALLBACK = {
    "Ko ndokkaami?":            "Alhamdulilahi, mi yiɗi waawi.",
    "Jam mi ne nde?":           "Jam, ko tummude jemma am.",
    "Nde a ne nde, baaba?":     "Alhamdulilahi, mi yiɗi waawi, ɓii am.",
    "Nde ko garko mbo a yiɗi?": "Garko mam ɓurta, cillanta e jowee mum.",
    "Ko heyde a toon nii?":     "Heyde moo aydu waali, suudu yiɗi jowaandi.",
    "Ko jaaji am nde waɗi?":    "Jaaji am ndoggi, ɓe yiɗi ndayrata.",
    "Nde ko mbarooga mbo?":     "Mbarooga mam jammu jammu, ɓe nasti ɓe yajje.",
    "Ko woli mbo a toon?":      "Woli moo toon: Gorko ɓii cuuɗa, ɓikkoy ɓe faamne.",
    "Nde mbarooga mbo?":        "Mbarooga mam jammu jammu, ɓe nasti ɓe yajje.",
}

def fallback(q):
    if q in FALLBACK:
        return FALLBACK[q]
    for k, v in FALLBACK.items():
        if any(w in q for w in k.split() if len(w) > 3):
            return v
    return random.choice(list(FALLBACK.values()))


# ══════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════
tok, model, model_label = load_model()

st.markdown(f"""
<div style="text-align:center;padding:10px 0 6px;border-bottom:2px solid #ffe0b2;margin-bottom:16px">
  <h1>🌿 Chatbot Pulaar</h1>
  <p style="color:#795548;font-style:italic;margin:0">Aadamu Arɗo — Berger / Sage du Fouta Toro</p>
  <p style="font-size:0.8em;color:#9e9e9e;margin:4px 0 0">Modèle : <b>{model_label}</b></p>
</div>
""", unsafe_allow_html=True)

if "fine-tuné" in model_label:
    st.success("✅ Modèle fine-tuné Pulaar chargé")
elif "pré-entraîné" in model_label:
    st.info("ℹ️ Modèle pré-entraîné (pas encore fine-tuné QA)")
else:
    st.warning("⚠️ GPT-2 brut — copiez `pulaar_model_finetuned/` ici pour activer le vrai modèle")

# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Génération")
    temperature = st.slider("Température",  0.3, 1.5, 0.7, 0.05)
    top_p       = st.slider("Top-p",        0.5, 1.0, 0.9, 0.05)
    max_new     = st.slider("Tokens max",   20,  150, 60,  10)

    st.markdown("---\n### 💬 Exemples")
    for ex in ["Ko ndokkaami?", "Nde ko garko mbo a yiɗi?",
               "Ko heyde a toon nii?", "Nde ko mbarooga mbo?",
               "Ko woli mbo a toon?", "Jam mi ne nde?"]:
        if st.button(ex, key=ex):
            st.session_state["pending"] = ex

    st.markdown("---")
    if st.button("🗑️ Effacer"):
        st.session_state["messages"] = []
        st.rerun()

    st.markdown("---\n### ℹ️ À propos")
    st.markdown("""
**Langue** : Pulaar (Fula/Fulfulde)  
**Modèle** : GPT-2 fine-tuné  
**Format** : `Q: ...\nR: ...`  
**Corpus** : FUB-Narratives  
**Catégories :** Salutations · Vie pastorale  
Famille · Contes & sagesse · Météo
    """)

# ── Messages ──────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "bot",
         "content": "Taalel taatel! Mi wiyee Aadamu Arɗo. Ko ndokkaami? 🌿",
         "mode": "accueil"}
    ]

for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f'<div class="msg-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        badge = f'<span class="badge">{msg.get("mode","")}</span>' if msg.get("mode") else ""
        st.markdown(f'<div class="msg-bot">🌿 {msg["content"]}{badge}</div>', unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────
pending    = st.session_state.pop("pending", None)
user_input = st.chat_input("Écrire en Pulaar… (ex: Ko ndokkaami?)")
query      = pending or user_input

if query and query.strip():
    q = query.strip()
    st.session_state["messages"].append({"role": "user", "content": q})

    with st.spinner("Aadamu Arɗo pense… 💭"):
        prompt = f"Q: {q}\nR:"
        try:
            answer = generate(prompt, model, tok, max_new, temperature, top_p)
            if not answer or len(answer) < 2:
                raise ValueError
            mode = model_label
        except Exception:
            answer = fallback(q)
            mode   = "fallback"

    st.session_state["messages"].append({"role": "bot", "content": answer, "mode": mode})
    st.rerun()

# ── Mode QA (Exercice 3) ──────────────────────────────
with st.expander("📚 Mode QA avec contexte (Exercice 3)", expanded=False):
    ctx = st.text_area("Contexte",
        "Pulaar woni demngal ngal ɓe poti e Mauritania, Senegaal, Gine, Mali e Niiseer.",
        height=80)
    qst = st.text_input("Question", "Nde demngal Pulaar hollata?")

    if st.button("🔍 Générer"):
        prompt_qa = f"Context: {ctx}\nQuestion: {qst}\nAnswer:"
        with st.spinner("Génération…"):
            try:
                ids = tok.encode(prompt_qa, return_tensors="pt")
                with torch.no_grad():
                    out = model.generate(ids, max_new_tokens=60,
                                         temperature=temperature, top_p=top_p,
                                         do_sample=True, pad_token_id=tok.eos_token_id)
                full = tok.decode(out[0], skip_special_tokens=True)
                rep  = full.split("Answer:", 1)[-1].strip().split("\n")[0]
            except Exception as e:
                rep = f"Erreur : {e}"
        st.success(f"**Réponse :** {rep}")
