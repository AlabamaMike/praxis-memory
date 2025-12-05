# PRAXIS System Specification

## Procedural Recall for Agents with eXperiences Indexed by State

**Version:** 1.0
**Based on:** arXiv:2511.22074v2 (Bi, Hu, and Nasir, 2025)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [System Overview](#3-system-overview)
4. [Core Concepts](#4-core-concepts)
5. [Architecture](#5-architecture)
6. [Memory Entry Structure](#6-memory-entry-structure)
7. [Retrieval Algorithm](#7-retrieval-algorithm)
8. [Integration Points](#8-integration-points)
9. [Configuration Parameters](#9-configuration-parameters)
10. [Data Flows](#10-data-flows)
11. [Performance Characteristics](#11-performance-characteristics)
12. [Evaluation Metrics](#12-evaluation-metrics)
13. [Future Extensions](#13-future-extensions)
14. [Glossary](#14-glossary)
15. [References](#15-references)

---

## 1. Executive Summary

PRAXIS is a lightweight post-training learning mechanism that enables AI agents to acquire procedural knowledge from experience in real-time. Unlike traditional approaches that rely on pre-defined Standard Operating Procedures (SOPs) or factual memory systems, PRAXIS stores the consequences of actions and retrieves them by jointly matching environmental and internal states from past episodes to the current state.

### Key Benefits

- **Improved Accuracy:** Average task completion increased from 40.3% to 44.1%
- **Enhanced Reliability:** Agent reliability improved from 74.5% to 79.0%
- **Greater Efficiency:** Steps-to-completion reduced from 25.2 to 20.2 on average
- **Model Agnostic:** Works across different foundation model backbones

### Design Principles

1. **State-Dependent Retrieval:** Memories are indexed and retrieved based on environmental and internal state similarity
2. **A Posteriori Learning:** Procedures are learned from experience rather than pre-specified
3. **Real-Time Operation:** Memory encoding and retrieval happen during agent execution
4. **Lightweight Integration:** Minimal modifications to existing agent architectures

---

## 2. Problem Statement

### 2.1 Types of Agent Knowledge

AI agents must learn two main classes of information:

| Type | Description | Characteristics | Examples |
|------|-------------|-----------------|----------|
| **Facts** | Atomic pieces of information | Context-independent, can change over time | User preferences, organizational charts, names |
| **Procedures** | Established conventions for doing things | State-dependent, sequence of requirements/preferences | Troubleshooting workflows, sales processes, form completion |

### 2.2 Limitations of Existing Approaches

**A Priori Procedural Specification (SOPs):**
- Many procedures are not fully documented
- Humans are often trained by observation, not documentation
- Enumerating all states and edge cases is combinatorially difficult
- Procedures become obsolete as environments change

**Existing Memory Frameworks:**
- RAG systems focus on factual knowledge retrieval
- Mem0 and Letta focus on long-term factual memory
- Reflexion and Self-Refine lack environment state encoding
- Workflow memory systems use high-level trajectories, not local state-action mappings

### 2.3 Target Environment

Web browsing is the primary target environment because:
- Requires multi-step procedural interactions
- Interfaces change frequently (seasonal pop-ups, redesigns)
- Procedures are rarely comprehensively documented
- High personalization limits pretraining coverage
- AI-generated interfaces create novel out-of-distribution states

---

## 3. System Overview

### 3.1 PRAXIS Definition

**PRAXIS** = **P**rocedural **R**ecall for **A**gents with e**X**periences **I**ndexed by **S**tate

A state-dependent memory system that:
1. **Stores** local interaction traces (state-action-result tuples)
2. **Retrieves** relevant memories by matching current state to past states
3. **Augments** action selection with retrieved exemplars

### 3.2 Theoretical Foundation

PRAXIS is inspired by **state-dependent memory** in psychology, which finds improved recall when:
- Internal state at retrieval matches internal state during encoding
- External context at retrieval matches external context during encoding

(Bower, 1981; Tulving & Thomson, 1973)

### 3.3 System Boundaries

```
┌─────────────────────────────────────────────────────────────────┐
│                        PRAXIS SYSTEM                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐ │
│  │  Memory Store   │◄───│ Memory Encoder  │◄───│ Experience  │ │
│  │                 │    │                 │    │   Stream    │ │
│  └────────┬────────┘    └─────────────────┘    └─────────────┘ │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │    Retriever    │───►│ Action Selector │                    │
│  │                 │    │   (Augmented)   │                    │
│  └─────────────────┘    └─────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Core Concepts

### 4.1 Environmental State (M^env)

The observable state of the environment at a given moment.

**In Web Browsing Context:**
- Current URL and page structure
- DOM elements and their properties
- Visual layout and rendered content
- Interactive elements available

**Representation:**
- Compressed textual description
- DOM feature set for IoU computation
- Visual features (optional)

### 4.2 Internal State (M^int)

The agent's internal context during experience.

**Components:**
- Current directive/goal being pursued
- Progress towards objective
- Relevant context from conversation/session

**Representation:**
- Natural language description
- Embedded vector for similarity computation

### 4.3 Action (a)

The action taken by the agent at a specific state.

**In Web Browsing Context:**
- Click actions (element, coordinates)
- Type/input actions (text, target field)
- Navigation actions (URL, back, forward)
- Scroll actions (direction, amount)
- Form submissions
- Custom agent actions

### 4.4 State Transition

A complete record of a state change:
```
(M^env-pre, M^int, a, M^env-post)
```

Where:
- `M^env-pre`: Environment state before action
- `M^int`: Internal state/objective during action
- `a`: Action taken
- `M^env-post`: Environment state after action

---

## 5. Architecture

### 5.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              AI AGENT SYSTEM                              │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                     SCAFFOLDING LAYER                              │ │
│  │                                                                    │ │
│  │   ┌─────────┐  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐ │ │
│  │   │ Input   │  │   State     │  │   Action     │  │   Output    │ │ │
│  │   │ Node    │──│ Assessment  │──│  Selection   │──│   Node      │ │ │
│  │   │         │  │   Node      │  │    Node      │  │             │ │ │
│  │   └─────────┘  └─────────────┘  └──────┬───────┘  └─────────────┘ │ │
│  │                                        │                          │ │
│  └────────────────────────────────────────┼──────────────────────────┘ │
│                                           │                            │
│  ┌────────────────────────────────────────┼──────────────────────────┐ │
│  │                    PRAXIS MEMORY SYSTEM │                          │ │
│  │                                        ▼                          │ │
│  │   ┌──────────────────────────────────────────────────────────┐   │ │
│  │   │              PROCEDURAL MEMORY CONTEXT                   │   │ │
│  │   │                                                          │   │ │
│  │   │  Retrieved Exemplars:                                    │   │ │
│  │   │  • (state₁, action₁, result₁)                           │   │ │
│  │   │  • (state₂, action₂, result₂)                           │   │ │
│  │   │  • ...                                                   │   │ │
│  │   └──────────────────────────────────────────────────────────┘   │ │
│  │                           ▲                                      │ │
│  │   ┌───────────────────────┴──────────────────────────────────┐   │ │
│  │   │                    RETRIEVAL ENGINE                      │   │ │
│  │   │                                                          │   │ │
│  │   │  Query: (Q^env, Q^int)                                   │   │ │
│  │   │                                                          │   │ │
│  │   │  1. Compute environment similarity (IoU + length)        │   │ │
│  │   │  2. Get top-k by environment score                       │   │ │
│  │   │  3. Re-rank by internal state similarity                 │   │ │
│  │   │  4. Filter by similarity threshold τ                     │   │ │
│  │   └──────────────────────────────────────────────────────────┘   │ │
│  │                           ▲                                      │ │
│  │   ┌───────────────────────┴──────────────────────────────────┐   │ │
│  │   │                    MEMORY STORE                          │   │ │
│  │   │                                                          │   │ │
│  │   │  Memory Entry i:                                         │   │ │
│  │   │  {                                                       │   │ │
│  │   │    M^env-pre_i : Environment state (pre-action)          │   │ │
│  │   │    M^int_i     : Internal state/objective                │   │ │
│  │   │    a_i         : Action taken                            │   │ │
│  │   │    M^env-post_i: Environment state (post-action)         │   │ │
│  │   │  }                                                       │   │ │
│  │   └──────────────────────────────────────────────────────────┘   │ │
│  └──────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ┌────────────────────────────────────────────────────────────────────┐│
│  │                    FOUNDATION MODEL (VLM)                          ││
│  │         (Llama 4, Qwen3-VL, Gemini 2.5, GPT-5, Claude, etc.)      ││
│  └────────────────────────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Node-Based Architecture

The agent uses a node-based scaffolding architecture where each node handles a specific function:

| Node | Function | PRAXIS Integration |
|------|----------|-------------------|
| Input Node | Receives environment observations | Source of M^env |
| State Assessment Node | Evaluates current state and progress | Source of M^int |
| **Action Selection Node** | Decides next action | **Receives procedural memory context** |
| Output Node | Executes actions in environment | Generates M^env-post |

### 5.3 Memory Subsystem Components

#### 5.3.1 Memory Encoder
- Converts raw experiences into structured memory entries
- Generates state descriptions from environment observations
- Captures internal state from agent context

#### 5.3.2 Memory Store
- Persists memory entries across sessions
- Supports efficient similarity search
- Maintains metadata for filtering and ranking

#### 5.3.3 Retrieval Engine
- Implements the retrieval algorithm (Algorithm 1)
- Computes similarity scores
- Returns ranked, filtered exemplars

#### 5.3.4 Context Injector
- Formats retrieved memories for the action selection node
- Manages context window budget
- Handles memory truncation if needed

---

## 6. Memory Entry Structure

### 6.1 Schema Definition

```typescript
interface MemoryEntry {
  // Unique identifier
  id: string;

  // Timestamp of creation
  created_at: timestamp;

  // Environment state before action
  env_state_pre: EnvironmentState;

  // Internal agent state
  internal_state: InternalState;

  // Action taken
  action: Action;

  // Environment state after action
  env_state_post: EnvironmentState;

  // Optional metadata
  metadata?: {
    task_id?: string;
    session_id?: string;
    success?: boolean;
    source?: 'agent' | 'demonstration';
  };
}

interface EnvironmentState {
  // Textual description
  description: string;

  // Feature set for IoU computation
  features: Set<string>;

  // Length metric (e.g., DOM element count)
  length: number;

  // Optional: URL or location identifier
  location?: string;

  // Optional: Visual embedding
  visual_embedding?: number[];
}

interface InternalState {
  // Natural language description of objective
  directive: string;

  // Embedding vector
  embedding: number[];

  // Optional: Progress indicator
  progress?: number;
}

interface Action {
  // Action type
  type: 'click' | 'type' | 'navigate' | 'scroll' | 'submit' | 'custom';

  // Action parameters
  params: Record<string, any>;

  // Natural language description
  description?: string;
}
```

### 6.2 Example Memory Entry

```json
{
  "id": "mem_abc123",
  "created_at": "2025-12-05T10:30:00Z",
  "env_state_pre": {
    "description": "Amazon product page for wireless headphones, showing product details, price $79.99, Add to Cart button visible, quantity selector showing 1",
    "features": ["amazon.com", "product-page", "add-to-cart-btn", "price-display", "quantity-selector", "product-title", "product-image"],
    "length": 156,
    "location": "https://amazon.com/dp/B09XYZ123"
  },
  "internal_state": {
    "directive": "Purchase wireless headphones with noise cancellation under $100",
    "embedding": [0.12, -0.45, 0.78, ...],
    "progress": 0.4
  },
  "action": {
    "type": "click",
    "params": {
      "element": "add-to-cart-button",
      "coordinates": [450, 320]
    },
    "description": "Click Add to Cart button"
  },
  "env_state_post": {
    "description": "Amazon cart confirmation modal showing 'Added to Cart' message, options to proceed to checkout or continue shopping",
    "features": ["amazon.com", "cart-modal", "checkout-btn", "continue-shopping-btn", "cart-confirmation"],
    "length": 42,
    "location": "https://amazon.com/dp/B09XYZ123"
  },
  "metadata": {
    "task_id": "task_purchase_headphones",
    "session_id": "sess_xyz789",
    "success": true,
    "source": "agent"
  }
}
```

---

## 7. Retrieval Algorithm

### 7.1 Algorithm Specification

```
Algorithm: Procedural Memory Retrieval

Input:
  - {M^env_i}^n_{i=1}  : Memory environment states
  - Q^env              : Query environment state
  - {M^int_i}^n_{i=1}  : Memory internal states
  - Q^int              : Query internal state
  - f                  : Internal state embedding function
  - k                  : Search breadth (top-k parameter)
  - τ                  : Similarity threshold

Output:
  - R                  : Retrieved memory indices (ordered by relevance)

Procedure:
  1. FOR i ← 1 to n DO:
     a. v_i ← IoU(M^env_i, Q^env)           // Feature overlap
     b. ℓ_m ← |M^env_i|                     // Memory state length
     c. ℓ_q ← |Q^env|                       // Query state length
     d. l_i ← LengthOverlap(ℓ_m, ℓ_q)       // Length similarity
     e. s^env_i ← v_i · l_i                 // Combined environment score
     f. s^int_i ← ⟨f(M^int_i), f(Q^int)⟩   // Internal state similarity

  2. R^env ← TopkIndices(s^env, k)          // Top-k by environment

  3. R̃ ← Sort(R^env, key=i ↦ s^int_i, desc) // Re-rank by internal

  4. R ← [i ∈ R̃ | s^env_i ≥ τ]             // Filter by threshold

  5. RETURN R
```

### 7.2 Similarity Functions

#### 7.2.1 Intersection over Union (IoU)

```
IoU(A, B) = |A ∩ B| / |A ∪ B|
```

Where A and B are feature sets representing environment states.

**Properties:**
- Range: [0, 1]
- Symmetric: IoU(A, B) = IoU(B, A)
- Identity: IoU(A, A) = 1

#### 7.2.2 Length Overlap

```
LengthOverlap(ℓ_m, ℓ_q) = 1 - |ℓ_m - ℓ_q| / max(ℓ_m, ℓ_q)
```

**Purpose:** Penalizes matches where state complexity differs significantly.

**Properties:**
- Range: [0, 1]
- Maximum when ℓ_m = ℓ_q
- Prevents matching simple states to complex ones

#### 7.2.3 Internal State Similarity

```
s^int = ⟨f(M^int), f(Q^int)⟩
```

Where f is an embedding function (e.g., sentence transformer) and ⟨·,·⟩ is cosine similarity.

### 7.3 Retrieval Pipeline Visualization

```
Query State                     Memory Store
┌─────────────────┐            ┌─────────────────────────────────┐
│ Q^env (current  │            │ Memory 1: (M^env_1, M^int_1, …) │
│ environment)    │            │ Memory 2: (M^env_2, M^int_2, …) │
│                 │            │ Memory 3: (M^env_3, M^int_3, …) │
│ Q^int (current  │            │ ...                             │
│ objective)      │            │ Memory n: (M^env_n, M^int_n, …) │
└────────┬────────┘            └────────────────┬────────────────┘
         │                                      │
         └──────────────┬───────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │   STEP 1: Environment Score  │
         │                              │
         │   For each memory i:         │
         │   s^env_i = IoU × LengthOvlp │
         └──────────────┬───────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │   STEP 2: Top-K Selection    │
         │                              │
         │   Select k memories with     │
         │   highest s^env scores       │
         └──────────────┬───────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │   STEP 3: Internal Re-rank   │
         │                              │
         │   Sort top-k by s^int        │
         │   (objective similarity)     │
         └──────────────┬───────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │   STEP 4: Threshold Filter   │
         │                              │
         │   Keep only where s^env ≥ τ  │
         └──────────────┬───────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │   Retrieved Memories R       │
         │                              │
         │   Ordered by relevance       │
         └──────────────────────────────┘
```

---

## 8. Integration Points

### 8.1 Memory Source Integration

PRAXIS supports memory population from multiple sources:

| Source | Description | Use Case |
|--------|-------------|----------|
| **Agent Trajectories** | Experiences generated by the agent during task execution | Continuous learning, self-improvement |
| **Human Demonstrations** | Expert-recorded interactions | Bootstrapping, quality exemplars |
| **Synthetic Generation** | Programmatically generated experiences | Coverage expansion, edge cases |

### 8.2 Action Selection Integration

The procedural memory context is injected into the action selection node's prompt:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ACTION SELECTION PROMPT                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [System Instructions]                                              │
│                                                                     │
│  [Current State Description]                                        │
│                                                                     │
│  [Task Objective]                                                   │
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ PROCEDURAL MEMORY CONTEXT                                     │ │
│  │                                                               │ │
│  │ The following past experiences may be relevant:               │ │
│  │                                                               │ │
│  │ Experience 1:                                                 │ │
│  │   State: [env_state_pre description]                         │ │
│  │   Action: [action description]                                │ │
│  │   Result: [env_state_post description]                       │ │
│  │                                                               │ │
│  │ Experience 2:                                                 │ │
│  │   State: [env_state_pre description]                         │ │
│  │   Action: [action description]                                │ │
│  │   Result: [env_state_post description]                       │ │
│  │                                                               │ │
│  │ ...                                                           │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  [Action Space Definition]                                          │
│                                                                     │
│  What action should be taken next?                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.3 Memory Persistence Layer

```
┌─────────────────────────────────────────────────────────────────┐
│                     PERSISTENCE OPTIONS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  In-Memory Store                                                 │
│  ├── Fast access, session-scoped                                │
│  └── Suitable for: Single session, testing                      │
│                                                                  │
│  Vector Database (e.g., Pinecone, Weaviate, Chroma)             │
│  ├── Efficient similarity search                                │
│  ├── Scalable to millions of memories                           │
│  └── Suitable for: Production, cross-session learning           │
│                                                                  │
│  Document Store (e.g., MongoDB, PostgreSQL)                     │
│  ├── Rich metadata queries                                      │
│  ├── Transactional guarantees                                   │
│  └── Suitable for: Analytics, auditing                          │
│                                                                  │
│  Hybrid (Vector + Document)                                      │
│  ├── Best of both worlds                                        │
│  └── Suitable for: Enterprise deployments                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Configuration Parameters

### 9.1 Core Parameters

| Parameter | Symbol | Type | Default | Description |
|-----------|--------|------|---------|-------------|
| Search Breadth | k | integer | 5-10 | Number of top environment matches to consider |
| Similarity Threshold | τ | float | 0.3-0.5 | Minimum environment similarity for inclusion |
| Max Retrieved | - | integer | 5 | Maximum memories returned to action selector |
| Embedding Dimension | - | integer | 768-1536 | Size of internal state embeddings |

### 9.2 Performance Tuning

| Parameter | Impact | Trade-off |
|-----------|--------|-----------|
| **Higher k** | More comprehensive search, better recall | Increased computation, potential noise |
| **Lower τ** | More memories retrieved | Risk of irrelevant matches |
| **Higher τ** | More precise matches | May miss useful experiences |
| **Larger embedding** | Richer semantic representation | Memory/compute cost |

### 9.3 Scaling Behavior

Based on ablation studies (Figure 3 in paper):

```
Performance vs Retrieval Breadth (k)

52% ┤                                    ●·····●
    │                              ●
50% ┤                    ●                    ●
    │               ●        ●
48% ┤          ●
    │
46% ┤     ·
    │    ●
44% ┤
    │  ·
42% ┤ ●
    │
40% ┼──┬───┬───┬───┬───┬───┬───┬───┬───┬───┬──
    0  1   2   3   4   5   6   7   8   9   10
                  Recall Breadth (k)
```

**Observations:**
- Performance generally increases with k
- Slight decreases within each step (local context crowding)
- Converges to plateau around k=8-10
- Diminishing returns beyond k=10

---

## 10. Data Flows

### 10.1 Memory Encoding Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                       MEMORY ENCODING FLOW                          │
└─────────────────────────────────────────────────────────────────────┘

  Agent Action Loop
        │
        ▼
┌───────────────────┐
│ 1. Observe State  │ ──────────────────────────────────────┐
│    (Pre-Action)   │                                       │
└────────┬──────────┘                                       │
         │                                                   │
         ▼                                                   │
┌───────────────────┐                                       │
│ 2. Select Action  │                                       │
└────────┬──────────┘                                       │
         │                                                   │
         ▼                                                   │
┌───────────────────┐                                       │
│ 3. Execute Action │                                       │
└────────┬──────────┘                                       │
         │                                                   │
         ▼                                                   │
┌───────────────────┐                                       │
│ 4. Observe State  │ ──────────────────────────────────────┤
│    (Post-Action)  │                                       │
└────────┬──────────┘                                       │
         │                                                   │
         │              ┌───────────────────────────────────┘
         │              │
         │              ▼
         │     ┌───────────────────┐
         │     │ 5. Create Memory  │
         │     │    Entry          │
         │     │                   │
         │     │  • env_state_pre  │
         │     │  • internal_state │
         │     │  • action         │
         │     │  • env_state_post │
         │     └────────┬──────────┘
         │              │
         │              ▼
         │     ┌───────────────────┐
         │     │ 6. Store Memory   │
         │     └───────────────────┘
         │
         ▼
   [Continue Loop]
```

### 10.2 Memory Retrieval Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                       MEMORY RETRIEVAL FLOW                         │
└─────────────────────────────────────────────────────────────────────┘

  Agent Needs to Select Action
        │
        ▼
┌───────────────────────────┐
│ 1. Capture Current State  │
│    • Q^env (environment)  │
│    • Q^int (objective)    │
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐     ┌──────────────────────┐
│ 2. Compute Environment    │◄────│ Memory Store         │
│    Similarity Scores      │     │ (all M^env entries)  │
└────────────┬──────────────┘     └──────────────────────┘
             │
             ▼
┌───────────────────────────┐
│ 3. Select Top-K by        │
│    Environment Score      │
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐     ┌──────────────────────┐
│ 4. Compute Internal State │◄────│ Embedding Function f │
│    Similarities for Top-K │     └──────────────────────┘
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐
│ 5. Re-rank by Internal    │
│    State Similarity       │
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐
│ 6. Apply Threshold Filter │
│    (s^env ≥ τ)            │
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐
│ 7. Format Retrieved       │
│    Memories for Prompt    │
└────────────┬──────────────┘
             │
             ▼
┌───────────────────────────┐
│ 8. Inject into Action     │
│    Selection Context      │
└───────────────────────────┘
```

---

## 11. Performance Characteristics

### 11.1 Benchmark Results (REAL Benchmark)

#### Mean Accuracy (5 repetitions)

| Model | Baseline | With PRAXIS | Improvement |
|-------|----------|-------------|-------------|
| Llama 4 | 25.2% | 28.9% | +3.7% |
| Qwen3-VL | 37.1% | 40.7% | +3.6% |
| Gemini 2.5 Flash | 42.0% | 48.6% | +6.6% |
| GPT-5 | 45.9% | 49.3% | +3.4% |
| Claude Sonnet 4.5 | 51.2% | 53.2% | +2.0% |
| **Average** | **40.3%** | **44.1%** | **+3.8%** |

#### Best-of-5 Accuracy

| Model | Baseline | With PRAXIS | Improvement |
|-------|----------|-------------|-------------|
| Llama 4 | 47.3% | 52.7% | +5.4% |
| Qwen3-VL | 44.6% | 47.3% | +2.7% |
| Gemini 2.5 Flash | 59.8% | 61.6% | +1.8% |
| GPT-5 | 56.2% | 57.1% | +0.9% |
| Claude Sonnet 4.5 | 60.7% | 59.8% | -0.9% |
| **Average** | **53.7%** | **55.7%** | **+2.0%** |

#### Reliability (Mean success rate on tasks with ≥1 success)

| Model | Baseline | With PRAXIS | Improvement |
|-------|----------|-------------|-------------|
| Llama 4 | 53.2% | 54.9% | +1.7% |
| Gemini 2.5 Flash | 70.1% | 78.8% | +8.7% |
| Qwen3-VL | 83.2% | 86.0% | +2.8% |
| GPT-5 | 81.6% | 86.2% | +4.6% |
| Claude Sonnet 4.5 | 84.4% | 89.0% | +4.6% |
| **Average** | **74.5%** | **79.0%** | **+4.5%** |

#### Efficiency (Average steps to completion)

| Model | Baseline | With PRAXIS | Reduction |
|-------|----------|-------------|-----------|
| Llama 4 | 19.8 | 16.2 | -3.6 |
| Qwen3-VL | 27.7 | 20.8 | -6.9 |
| Gemini 2.5 Flash | 28.9 | 22.3 | -6.6 |
| GPT-5 | 24.2 | 20.7 | -3.5 |
| Claude Sonnet 4.5 | 25.2 | 21.0 | -4.2 |
| **Average** | **25.2** | **20.2** | **-5.0** |

### 11.2 Performance Analysis

**Key Findings:**
1. PRAXIS provides consistent improvements across all tested VLM backbones
2. Larger gains observed for mid-tier models (Gemini 2.5 Flash)
3. Reliability improvements are substantial (+4.5% average)
4. Efficiency gains indicate more direct trajectories (-20% average steps)
5. Generalization to unseen but similar tasks is demonstrated

---

## 12. Evaluation Metrics

### 12.1 Primary Metrics

| Metric | Definition | Purpose |
|--------|------------|---------|
| **Task Accuracy** | % of tasks completed successfully | Overall effectiveness |
| **Best-of-N Accuracy** | % with at least one success in N attempts | Capability frontier |
| **Reliability** | Mean success rate over repeated attempts | Consistency measure |
| **Steps to Completion** | Average actions to complete task | Efficiency measure |

### 12.2 Memory-Specific Metrics

| Metric | Definition | Purpose |
|--------|------------|---------|
| **Retrieval Precision** | % retrieved memories that were useful | Quality of retrieval |
| **Memory Utilization** | % of retrieved memories that influenced action | Impact assessment |
| **Coverage** | % of encountered states with relevant memories | Memory completeness |
| **Staleness** | Age of retrieved memories | Freshness tracking |

### 12.3 Evaluation Protocol

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION PROTOCOL                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. BENCHMARK SELECTION                                         │
│     • Use deterministic environment (e.g., REAL benchmark)      │
│     • Ensure reproducible task definitions                      │
│     • Include variety of task complexities                      │
│                                                                 │
│  2. BASELINE ESTABLISHMENT                                      │
│     • Run agent without PRAXIS memory                           │
│     • Multiple repetitions (≥5) per task                        │
│     • Record all metrics                                        │
│                                                                 │
│  3. PRAXIS EVALUATION                                           │
│     • Enable PRAXIS memory system                               │
│     • Same tasks, same number of repetitions                    │
│     • Allow memory accumulation across runs                     │
│                                                                 │
│  4. ABLATION STUDIES                                            │
│     • Vary k (retrieval breadth)                                │
│     • Vary τ (similarity threshold)                             │
│     • Test different embedding functions                        │
│                                                                 │
│  5. GENERALIZATION TESTING                                      │
│     • Train on subset of tasks                                  │
│     • Evaluate on held-out tasks                                │
│     • Measure transfer to similar environments                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 13. Future Extensions

### 13.1 Beyond Web Environments

PRAXIS is environment-agnostic and can extend to:
- Desktop application automation
- Mobile app interactions
- API orchestration
- Robotic manipulation
- Game playing

### 13.2 Richer State Encoding

Current implementation uses basic features. Enhancements include:
- **Visual Encoders:** Pre-trained vision models for screenshot understanding
- **Semantic DOM Parsing:** Structure-aware element representations
- **Temporal Context:** Sequence models for action history
- **Multi-Modal Fusion:** Combined visual + textual + structural features

### 13.3 Adaptive Retrieval Mechanisms

Instead of fixed similarity heuristics:
- **Uncertainty-Based:** Retrieve more when model is uncertain
- **Budget-Aware:** Adapt to available compute/context window
- **Iterative Refinement:** Multiple retrieval rounds for complex states
- **Learned Retrieval:** Neural retrieval functions trained on outcomes

### 13.4 From Action Agents to Alignment Agents

Extend training signal beyond task success:
- **User Preference Learning:** Observe user feedback over time
- **Style Adaptation:** Learn how a user prefers tasks to be done
- **Preference Convergence:** Build procedural memory of user preferences
- **Personalization:** Per-user memory stores for customized behavior

### 13.5 Memory Management

- **Memory Consolidation:** Merge similar experiences
- **Forgetting Mechanisms:** Decay unused or outdated memories
- **Quality Scoring:** Prioritize high-quality demonstrations
- **Conflict Resolution:** Handle contradictory memories

---

## 14. Glossary

| Term | Definition |
|------|------------|
| **A Posteriori Learning** | Learning from experience after deployment, rather than pre-specification |
| **A Priori Specification** | Pre-defining procedures before deployment (e.g., SOPs) |
| **DOM** | Document Object Model - tree structure representing web page elements |
| **Environmental State** | Observable state of the external environment |
| **Internal State** | Agent's internal context including objectives and progress |
| **IoU** | Intersection over Union - similarity metric for set comparison |
| **Node-Based Architecture** | Agent design with specialized components (nodes) for different functions |
| **PRAXIS** | Procedural Recall for Agents with eXperiences Indexed by State |
| **Procedural Knowledge** | Knowledge about how to do things (vs. factual knowledge about what things are) |
| **RAG** | Retrieval-Augmented Generation |
| **REAL Benchmark** | Benchmark with deterministic clones of functional websites |
| **Retrieval Breadth (k)** | Number of top candidates considered during memory retrieval |
| **Similarity Threshold (τ)** | Minimum similarity score required for memory inclusion |
| **SOP** | Standard Operating Procedure |
| **State-Dependent Memory** | Psychological concept where recall improves when retrieval context matches encoding context |
| **VLM** | Vision-Language Model |

---

## 15. References

1. Altrina. 2025. "Evolving Our State-of-the-Art Browsing Agent."
2. Altrina. 2025. "Introducing Large Neurosymbolic Cognitive Models."
3. Bower, G.H. 1981. "Mood and memory." American Psychologist.
4. Chhikara et al. 2025. "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory."
5. Deng et al. 2023. "Mind2Web: Towards a Generalist Agent for the Web."
6. Garg et al. 2025. "REAL: Benchmarking Autonomous Agents on Deterministic Simulations of Real Websites."
7. He et al. 2024. "WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models."
8. Lewis et al. 2021. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks."
9. Liu et al. 2018. "Reinforcement Learning on Web Interfaces Using Workflow-Guided Exploration."
10. Madaan et al. 2023. "Self-Refine: Iterative Refinement with Self-Feedback."
11. Majumder et al. 2023. "CLIN: A Continually Learning Language Agent for Rapid Task Adaptation and Generalization."
12. Packer et al. 2024. "MemGPT: Towards LLMs as Operating Systems."
13. Shinn et al. 2023. "Reflexion: Language Agents with Verbal Reinforcement Learning."
14. Tulving & Thomson. 1973. "Encoding specificity and retrieval processes in episodic memory."
15. Wang et al. 2024. "Agent Workflow Memory."
16. Xu et al. 2025. "A-MEM: Agentic Memory for LLM Agents."
17. Zhao et al. 2023. "ExpeL: LLM Agents Are Experiential Learners."
18. Zheng et al. 2024. "Synapse: Trajectory-as-Exemplar Prompting with Memory for Computer Control."
19. Zhong et al. 2023. "MemoryBank: Enhancing Large Language Models with Long-Term Memory."
20. Zhou et al. 2024. "WebArena: A Realistic Web Environment for Building Autonomous Agents."

---

*This specification is based on arXiv:2511.22074v2 by Dasheng Bi, Yubin Hu, and Mohammed N. Nasir (Altrina, 2025).*
