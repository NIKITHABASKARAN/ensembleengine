# Luminaries Zero Trust PDP - Security Dashboard Specification

## Project Overview

**Project Name:** Luminaries Zero Trust PDP Security Dashboard  
**Project Type:** React Web Application (Single Page Application)  
**Core Functionality:** A professional security dashboard that visualizes ML model predictions for access control decisions, featuring real-time simulation controls, live event feeds, and risk visualization charts.  
**Target Users:** Security analysts, judges, and stakeholders evaluating the ML-based Zero Trust Policy Decision Point.

---

## 1. Technical Stack

| Component | Technology |
|-----------|------------|
| Frontend Framework | React 18+ with Vite |
| Styling | Tailwind CSS |
| Charts/Visualizations | Recharts |
| HTTP Client | Axios |
| State Management | React hooks (useState, useEffect) |
| Backend | FastAPI (existing at localhost:8000) |

---

## 2. UI/UX Specification

### 2.1 Layout Structure

```
┌─────────────────────────────────────────────────────────────────────┐
│                         HEADER                                      │
│              "Luminaries Zero Trust PDP"                            │
│                    Atos Logo + Status                               │
├──────────┬──────────────────────────────────────────────────────────┤
│          │                    MAIN CONTENT                         │
│  SIDEBAR │  ┌─────────────────────┐  ┌────────────────────────┐   │
│          │  │   EVENT FEED        │  │  RISK FUSION RADAR     │   │
│  [Normal]│  │   (Live Table)      │  │  (Recharts Radar)      │   │
│          │  │                     │  │                        │   │
│  [Cred   │  │   User ID            │  │  lightgbm              │   │
│  Stuffing]│  │   Resource Path     │  │  xgboost               │   │
│          │  │   Action            │  │  isolation             │   │
│  [Impos- │  │   Verdict Badge     │  │  lstm                  │   │
│  sible   │  │                     │  │  gnn                   │   │
│  Travel] │  └─────────────────────┘  └────────────────────────┘   │
│          │  ┌─────────────────────┐  ┌────────────────────────┐   │
│  [Insider│  │  UNCERTAINTY GAUGE  │  │  CUSTOM SIMULATION     │   │
│  Threat] │  │  (Circular Gauge)   │  │  (Form Builder)        │   │
│          │  │  Entropy: X.XX      │  │                        │   │
│          │  │  [HIGH UNCERTAINTY] │  │                        │   │
│          │  └─────────────────────┘  └────────────────────────┘   │
└──────────┴──────────────────────────────────────────────────────────┘
```

### 2.2 Responsive Breakpoints

| Breakpoint | Width | Layout Changes |
|------------|-------|----------------|
| Desktop | ≥1280px | Full sidebar + 2-column grid |
| Tablet | 768-1279px | Collapsible sidebar + stacked cards |
| Mobile | <768px | Hidden sidebar (hamburger) + single column |

### 2.3 Visual Design

#### Color Palette (Atos Theme)

| Role | Color | Hex Code |
|------|-------|----------|
| Background Primary | Mine Shaft | `#2B2B2B` |
| Background Secondary | Dark Charcoal | `#1E1E1E` |
| Background Card | Charcoal | `#363636` |
| Primary/Accent | Azure Radiance | `#0073E6` |
| Primary Hover | Azure Dark | `#005BB5` |
| Text Primary | White | `#FFFFFF` |
| Text Secondary | Silver | `#B0B0B0` |
| Text Muted | Gray | `#808080` |
| Success/ALLOW | Emerald | `#10B981` |
| Warning/MFA | Amber | `#F59E0B` |
| Danger/BLOCK | Red | `#EF4444` |
| Border | Dark Gray | `#404040` |

#### Typography

| Element | Font | Size | Weight |
|---------|------|------|--------|
| Header Title | Inter | 24px | 700 |
| Section Headers | Inter | 18px | 600 |
| Body Text | Inter | 14px | 400 |
| Table Data | JetBrains Mono | 13px | 400 |
| Badges | Inter | 12px | 600 |
| Labels | Inter | 12px | 500 |

#### Spacing System

- Base unit: 4px
- Component padding: 16px (4 units)
- Card margin: 16px
- Section gap: 24px
- Sidebar width: 240px

#### Visual Effects

- Card shadows: `0 4px 6px -1px rgba(0, 0, 0, 0.3)`
- Border radius: 12px (cards), 8px (buttons), 6px (badges)
- Transitions: 200ms ease-in-out for all interactive elements
- Glassmorphism on cards: `backdrop-filter: blur(8px)`

---

## 3. Component Specifications

### 3.1 Header Component

- **Height:** 64px
- **Background:** `#1E1E1E` with bottom border `#404040`
- **Content:**
  - Left: Atos-style shield icon + "Luminaries Zero Trust PDP" title
  - Right: Connection status indicator (green dot = connected, red = disconnected)
  - Right: Model status badges showing loaded models

### 3.2 Sidebar Component

- **Width:** 240px fixed
- **Background:** `#1E1E1E`
- **Border Right:** 1px solid `#404040`

#### Simulation Buttons (4 total)

| Button Label | Scenario ID | Visual Style |
|--------------|-------------|--------------|
| Normal | `Normal_Baseline` | Default: `#363636`, Active: `#0073E6` border |
| Credential Stuffing | `Credential_Stuffing` | Same |
| Impossible Travel | `Impossible_Travel` | Same |
| Insider Threat | `Insider_Threat` | Same |

**Button States:**
- Default: Background `#363636`, Text `#B0B0B0`
- Hover: Background `#404040`, Text `#FFFFFF`
- Active/Selected: Background `#0073E6`, Text `#FFFFFF`, glow effect

### 3.3 Event Feed (Live Table)

- **Container:** Card with `#363636` background
- **Columns:**
  1. Timestamp (sortable)
  2. User ID
  3. Resource Path
  4. Action Type
  5. Verdict Badge

#### Verdict Badge Colors

| Verdict | Background | Text | Icon |
|---------|------------|------|------|
| ALLOW | `#10B981` (20% opacity) | `#10B981` | ✓ check |
| MFA | `#F59E0B` (20% opacity) | `#F59E0B` | ⚠ warning |
| BLOCK | `#EF4444` (20% opacity) | `#EF4444` | ✕ cross |

- **Row hover:** Background `#404040`
- **Max visible rows:** 10 (scrollable)
- **Auto-scroll:** New events appear at top with animation

### 3.4 Risk Fusion Radar Chart

- **Library:** Recharts `RadarChart`
- **Dimensions:** 400x400px
- **Data Points:** 5 axes representing model probabilities
  - `lightgbm` - LightGBM model probability
  - `xgboost` - XGBoost model probability
  - `isolation` - Isolation Forest probability
  - `lstm` - LSTM sequential anomaly score
  - `gnn` - GraphSAGE GNN score
- **Range:** 0 to 1 (probability)
- **Colors:**
  - Fill: `#0073E6` at 30% opacity
  - Stroke: `#0073E6`
  - Grid: `#404040`
- **Animation:** Smooth entrance animation on data update

### 3.5 Uncertainty Gauge (Circular)

- **Type:** Custom SVG circular progress
- **Size:** 180px diameter
- **Value Range:** 0.0 to 1.0
- **Visual:**
  - Background ring: `#404040`
  - Progress ring: Gradient from `#10B981` (low) → `#F59E0B` (medium) → `#EF4444` (high)
  - Center: Large entropy value display
- **Warning State:**
  - When entropy > 0.7: Show "HIGH UNCERTAINTY" badge with pulsing animation
  - Badge: Red background with white text

### 3.6 Custom Simulation Form

- **Fields:**
  - User ID (text input)
  - Resource Path (text input)
  - Action Type (dropdown: READ, WRITE, DELETE, EXECUTE, ADMIN)
  - Device Type (dropdown: desktop, mobile, tablet, bot, unknown)
  - Country (dropdown with common countries)
  - Sensitivity Level (slider: 1-5)
  - Login Successful (toggle)
  - Is Attack IP (toggle)
- **Submit Button:** "Run Prediction" - Primary blue `#0073E6`
- **Response Display:** Shows full prediction result in expandable JSON viewer

### 3.7 MFA Modal Popup

- **Trigger:** When API returns `verdict: "MFA"`
- **Overlay:** Black at 60% opacity
- **Modal Card:**
  - Background: `#363636`
  - Border: 2px solid `#F59E0B`
  - Width: 400px
- **Content:**
  - Icon: Shield with warning
  - Title: "MFA Verification Required"
  - Description: "This access request requires additional verification"
  - Button: "Simulate Biometric Check" (triggers 2-second loading, then grants access)
- **Animation:** Fade in + scale up

---

## 4. Functionality Specification

### 4.1 Backend API Integration

#### Base URL
```
http://localhost:8000
```

#### Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Check backend and model status |
| POST | `/predict` | Submit single access request for evaluation |
| POST | `/simulate` | Trigger scenario-based simulation |

#### Simulation Scenarios

```javascript
const SCENARIOS = {
  "Normal_Baseline": "Healthy low-risk traffic",
  "Credential_Stuffing": "Attack pattern from malicious IPs",
  "Impossible_Travel": "Same user from two continents",
  "Insider_Threat": "Trusted employee at unusual hours"
};
```

### 4.2 Polling Mechanism

- **Poll Interval:** 3 seconds
- **Endpoints Polled:** `/simulate` (when scenario active)
- **Real-time Updates:** New events pushed to event feed
- **Manual Refresh:** Optional "Refresh" button

### 4.3 Data Flow

```
User clicks scenario → POST /simulate → Backend processes
                        ↓
                 ML Models Run (LightGBM, XGBoost, Isolation, LSTM, GNN)
                        ↓
                 Ensemble Score + Verdict
                        ↓
                 Frontend receives → Update Event Feed + Radar + Gauge
```

### 4.4 State Management

```typescript
interface DashboardState {
  // Connection
  isConnected: boolean;
  modelsLoaded: ModelStatus[];
  
  // Events
  events: SecurityEvent[];
  
  // Current Analysis
  lastPrediction: PredictionResult | null;
  
  // UI State
  activeScenario: string | null;
  isModalOpen: boolean;
  pendingMFAEvent: SecurityEvent | null;
}
```

### 4.5 Edge Cases

| Scenario | Handling |
|----------|----------|
| Backend disconnected | Show "Disconnected" status, disable buttons, retry every 5s |
| API timeout | Show error toast, keep previous data |
| MFA modal - user closes | Keep event in "MFA Pending" state |
| Empty event feed | Show placeholder "No events yet. Run a simulation!" |
| High entropy warning | Pulse animation on gauge, show warning badge |

---

## 5. API Response Schema

### 5.1 `/simulate` Response

```json
{
  "scenario": "Credential_Stuffing",
  "description": "Many users, many IPs, rapid failures",
  "events_sent": 5,
  "results": [
    {
      "event_id": "uuid",
      "user_id": "user_1234",
      "result": {
        "ensemble": {
          "risk_score": 0.85,
          "prediction": 1,
          "risk_level": "High",
          "verdict": "BLOCK"
        },
        "models": {
          "lightgbm": { "probability": 0.82, "weight": 0.45 },
          "xgboost": { "probability": 0.78, "weight": 0.35 },
          "elliptic_envelope": { "risk_probability": 0.65, "weight": 0.20 }
        },
        "deep_path": {
          "triggered": true,
          "lstm_sequential": { "risk_score": 0.72 },
          "gnn_link_predictor": { "risk_score": 0.68 }
        },
        "entropy": 0.45,
        "verdict": "BLOCK"
      }
    }
  ]
}
```

### 5.2 `/predict` Response

```json
{
  "ensemble": {
    "risk_score": 0.55,
    "prediction": 1,
    "risk_level": "Medium",
    "verdict": "MFA"
  },
  "models": { ... },
  "deep_path": { ... },
  "entropy": 0.72,
  "verdict": "MFA",
  "travel": { "flagged": false },
  "opa_reason": "fallback_mfa_moderate_risk"
}
```

---

## 6. File Structure

```
dashboard/
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
├── postcss.config.js
├── public/
│   └── favicon.svg
└── src/
    ├── main.jsx
    ├── App.jsx
    ├── index.css
    ├── components/
    │   ├── Header.jsx
    │   ├── Sidebar.jsx
    │   ├── EventFeed.jsx
    │   ├── RiskRadar.jsx
    │   ├── UncertaintyGauge.jsx
    │   ├── CustomSimulation.jsx
    │   └── MFAModal.jsx
    ├── hooks/
    │   └── usePolling.js
    ├── services/
    │   └── api.js
    └── utils/
        └── formatters.js
```

---

## 7. Acceptance Criteria

### Visual Checkpoints

- [ ] Header displays with Atos branding and connection status
- [ ] Sidebar shows 4 scenario buttons with correct styling
- [ ] Event feed displays rows with color-coded verdict badges
- [ ] Radar chart shows 5 model axes with proper labels
- [ ] Uncertainty gauge displays circular progress with value
- [ ] High entropy warning appears when entropy > 0.7
- [ ] Custom simulation form has all specified fields
- [ ] MFA modal appears with correct styling and behavior

### Functional Checkpoints

- [ ] Clicking "Normal" triggers /simulate with Normal_Baseline
- [ ] Clicking "Credential Stuffing" triggers attack simulation
- [ ] Event feed updates in real-time with new events
- [ ] Radar chart updates when new prediction arrives
- [ ] Gauge updates with current entropy value
- [ ] Custom form submits to /predict and displays result
- [ ] MFA modal triggers on verdict: "MFA"
- [ ] Application handles backend disconnection gracefully

### Theme Compliance

- [ ] Background uses #2B2B2B (Mine Shaft)
- [ ] Primary accent uses #0073E6 (Azure Radiance)
- [ ] All text is readable on dark background
- [ ] Cards have proper glassmorphism effect
- [ ] Animations are smooth (200ms transitions)

---

## 8. Running the Application

### Prerequisites

1. Start the backend:
   ```bash
   cd /path/to/ensembleengine
   uvicorn app_unified:fastapi_app --host 0.0.0.0 --port 8000
   ```

2. Start the frontend:
   ```bash
   cd dashboard
   npm install
   npm run dev
   ```

3. Access dashboard at: `http://localhost:5173`

---

*Specification Version: 1.0*  
*Created: 2026-03-22*  
*For: Luminaries Zero Trust PDP Demo*
