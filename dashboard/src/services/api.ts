import { PersonaType, NeuralAnalysisResponse, SecurityEvent } from '../types';

const API_BASE_URL = 'http://localhost:8000';

// Map persona types to backend scenario IDs
const personaToScenario: Record<PersonaType, string> = {
  normal: 'Normal_Baseline',
  impossible_travel: 'Impossible_Travel',
  credential_stuffing: 'Credential_Stuffing',
  insider_threat: 'Insider_Threat',
};

export async function analyzeSecurityEvent(
  persona: PersonaType,
  userId: string = 'user_demo'
): Promise<NeuralAnalysisResponse> {
  try {
    // Call the simulate endpoint with the scenario
    const response = await fetch(`${API_BASE_URL}/simulate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        scenario_id: personaToScenario[persona],
        count: 1,
      }),
    });

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`);
    }

    const data = await response.json();
    
    if (data.results && data.results.length > 0) {
      const result = data.results[0].result;
      return transformBackendResponse(result, persona);
    }
    
    return generateMockAnalysis(persona);
  } catch (error) {
    console.error('API Error:', error);
    return generateMockAnalysis(persona);
  }
}

// Transform backend response to our frontend format
function transformBackendResponse(result: any, persona: PersonaType): NeuralAnalysisResponse {
  const ensemble = result.ensemble || {};
  const models = result.models || {};
  const deepPath = result.deep_path || {};
  const input = result.input_summary || {};
  
  const riskScore = ensemble.risk_score || 0;
  const entropy = result.entropy || 0;
  
  // Map backend verdict to frontend
  let verdict: 'ALLOW' | 'MFA' | 'DENY';
  switch (result.verdict) {
    case 'ALLOW':
      verdict = 'ALLOW';
      break;
    case 'MFA':
      verdict = 'MFA';
      break;
    case 'BLOCK':
      verdict = 'DENY';
      break;
    default:
      verdict = riskScore > 0.7 ? 'DENY' : riskScore > 0.3 ? 'MFA' : 'ALLOW';
  }

  // Determine processing stage
  let processingStage: 'fast_path' | 'deep_path' | 'opa_enforcement';
  if (deepPath.triggered) {
    processingStage = 'deep_path';
  } else if (verdict === 'DENY') {
    processingStage = 'opa_enforcement';
  } else {
    processingStage = 'fast_path';
  }

  return {
    verdict,
    risk_score: riskScore,
    entropy,
    ensemble_weights: {
      lgbm: models.lightgbm?.weight || 0.45,
      xgb: models.xgboost?.weight || 0.35,
      iso: models.elliptic_envelope?.weight || 0.20,
      lstm: deepPath.triggered ? 0.15 : 0,
      gnn: deepPath.triggered ? 0.15 : 0,
    },
    deep_path: deepPath.triggered || false,
    behavioral_analysis: {
      typing_cadence: persona === 'credential_stuffing' ? 0.05 : 0.65 + Math.random() * 0.2,
      mouse_velocity: persona === 'credential_stuffing' ? 0.02 : 0.55 + Math.random() * 0.3,
      session_duration: persona === 'insider_threat' ? 14580 : 1200 + Math.random() * 3600,
      resource_pattern: persona === 'insider_threat' ? 'data_exfiltration' : 'normal',
    },
    travel_anomaly: result.travel?.flagged ? {
      detected: true,
      from_location: 'Previous Location',
      to_location: input.country || 'Unknown',
      velocity_kph: result.travel.velocity_kmh,
      time_delta_hours: 0.5,
    } : undefined,
    explanation: getExplanation(persona, riskScore, verdict),
    processing_stage: processingStage,
  };
}

function getExplanation(persona: PersonaType, riskScore: number, verdict: string): string {
  const explanations: Record<PersonaType, string> = {
    normal: 'All behavioral patterns within expected parameters. User authentication approved via fast path processing.',
    impossible_travel: 'Geographic anomaly detected: User accessed system from two distant locations within physically impossible timeframe. Deep path analysis triggered, MFA required.',
    credential_stuffing: 'Automated behavior detected: Typing cadence and mouse velocity patterns inconsistent with human interaction. Multiple failed login attempts. MFA verification required.',
    insider_threat: 'High-risk activity pattern: Extended session duration with unusual resource access patterns suggesting potential data exfiltration. OPA policy enforcement initiated - access denied.',
  };
  return explanations[persona];
}

export function generateMockAnalysis(persona: PersonaType): NeuralAnalysisResponse {
  const riskScores: Record<PersonaType, number> = {
    normal: 0.15,
    impossible_travel: 0.85,
    credential_stuffing: 0.75,
    insider_threat: 0.92,
  };

  const verdicts: Record<PersonaType, 'ALLOW' | 'MFA' | 'DENY'> = {
    normal: 'ALLOW',
    impossible_travel: 'MFA',
    credential_stuffing: 'MFA',
    insider_threat: 'DENY',
  };

  const stages: Record<PersonaType, 'fast_path' | 'deep_path' | 'opa_enforcement'> = {
    normal: 'fast_path',
    impossible_travel: 'deep_path',
    credential_stuffing: 'deep_path',
    insider_threat: 'opa_enforcement',
  };

  const riskScore = riskScores[persona];

  return {
    verdict: verdicts[persona],
    risk_score: riskScore,
    entropy: persona === 'normal' ? 0.12 : persona === 'impossible_travel' ? 0.45 : persona === 'credential_stuffing' ? 0.58 : 0.78,
    ensemble_weights: {
      lgbm: 0.25 + Math.random() * 0.1,
      xgb: 0.25 + Math.random() * 0.1,
      iso: 0.15 + Math.random() * 0.1,
      lstm: 0.2 + Math.random() * 0.1,
      gnn: 0.15 + Math.random() * 0.1,
    },
    deep_path: persona !== 'normal',
    behavioral_analysis: {
      typing_cadence: persona === 'credential_stuffing' ? 0.05 : 0.65 + Math.random() * 0.2,
      mouse_velocity: persona === 'credential_stuffing' ? 0.02 : 0.55 + Math.random() * 0.3,
      session_duration: persona === 'insider_threat' ? 14580 : 1200 + Math.random() * 3600,
      resource_pattern: persona === 'insider_threat' ? 'data_exfiltration' : 'normal',
    },
    travel_anomaly: persona === 'impossible_travel' ? {
      detected: true,
      from_location: 'New York, USA',
      to_location: 'Tokyo, Japan',
      velocity_kph: 42000,
      time_delta_hours: 0.5,
    } : undefined,
    explanation: getExplanation(persona, riskScore, verdicts[persona]),
    processing_stage: stages[persona],
  };
}

function generateResourceName(): string {
  const resources = [
    '/api/v1/users',
    '/api/v1/documents',
    '/api/v1/financial/reports',
    '/api/v1/admin/settings',
    '/api/v1/data/export',
  ];
  return resources[Math.floor(Math.random() * resources.length)];
}

function generateActionType(): string {
  const actions = ['read', 'write', 'delete', 'export', 'admin_access'];
  return actions[Math.floor(Math.random() * actions.length)];
}

export function createSecurityEvent(
  analysis: NeuralAnalysisResponse,
  persona: PersonaType,
  userId: string
): SecurityEvent {
  return {
    id: `evt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    timestamp: new Date().toLocaleTimeString(),
    user_id: userId,
    resource: generateResourceName(),
    action: generateActionType(),
    risk_score: analysis.risk_score,
    verdict: analysis.verdict,
    persona: persona,
  };
}