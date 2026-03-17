Design a clean, minimal, and visually striking web application UI for a Retrieval-Augmented Generation (RAG) system.

Theme: "Neural Observatory"

Concept:
The interface should feel like observing an AI system thinking in real time. It should be elegant, slightly futuristic, and data-centric, while remaining simple and easy to implement.

Core Layout:
Use a centered layout with three main sections:

1. Input area (top)
2. Answer display (center)
3. Reasoning / pipeline panel (right side or collapsible)

---

Input Area:

* A rounded text input box for user questions
* A primary button labeled “Ask”
* Subtle glow or focus effect when active

---

Answer Display (with Typewriter Effect):

* The answer should appear progressively using a typewriter-style effect (word-by-word or character-by-character)
* Include a blinking cursor at the end of the text
* Smooth, readable text rendering (no abrupt jumps)
* While generating:

  * show subtle animation (cursor blinking, soft glow)
  * display status text like “Generating answer...”
* After generation completes:

  * cursor becomes subtle or stops blinking
  * hallucination score fades in

---

Hallucination Score:

* Display as a colored badge and a horizontal progress bar (0.0–1.0)
* Color coding:

  * Green (>= 0.7): grounded
  * Yellow (0.4–0.7): partially grounded
  * Red (< 0.4): low confidence / hallucinated
* Include a short explanation of the score

---

Reasoning / Pipeline Panel:

* Designed as a vertical timeline or step tracker
* Show stages:

  * Query Rewrite
  * Retrieval
  * Generation
  * Hallucination Detection
* Active stage glows (blue)
* Completed stages are dimmed or green
* Each stage can expand to show details

Details to display:

* Rewritten query
* Retrieved document snippets (with labels like “Doc 1”, “Page 3”)
* Highlight supporting text where possible
* Unsupported claims if detected

---

Retry Behavior:

* If hallucination score is below threshold:

  * show “Retrying due to low grounding score”
  * clear or fade out previous answer
  * restart typewriter animation
* Show attempt count (e.g., Attempt 1, Attempt 2)

---

Interaction States:

* Loading: show step-by-step progress (rewriting → retrieval → generation → grading)
* Generating: typewriter animation active
* Retry: visible transition and message
* Final: stable view with answer + score + full reasoning
* Error: minimal error banner (e.g., connection lost, no documents found)

---

Visual Style:

* Dark mode primary theme
* Background: deep charcoal or dark navy (#0B0F14)
* Accent colors:

  * Electric blue (active processes)
  * Soft violet (reasoning)
  * Green (successful grounding)
  * Orange/red (warnings)
* Use subtle gradients and soft glow effects (avoid heavy neon)
* Typography: modern sans-serif (Inter or similar)
* Use cards, spacing, and light glassmorphism for structure

---

Constraints:

* Keep UI simple and implementable (no heavy animations or complex 3D effects)
* Do not include multi-hop reasoning visualization
* Do not include named entity tracking or consistency graphs
* Avoid clutter and unnecessary features

---

Goal:
Create a clean, interactive interface that clearly shows the RAG pipeline and makes the answer generation feel alive through typewriter-style streaming, while remaining minimal, understandable, and suitable for a student project implementation.
