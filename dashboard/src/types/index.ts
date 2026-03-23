export type PersonaType = 'normal' | 'impossible_travel' | 'credential_stuffing' | 'insider_threat';

export interface EnsembleWeights {
  lgbm: number;
  xgb: number;
  iso: number;
  lstm: number;
  gnn: number;
}

export interface BehavioralAnalysis {
  typing_cadence: number;
  mouse_velocity: number;
  session_duration: number;
  resource_pattern: string;
}

export interface TravelAnomaly {
  detected: boolean;
  from_location?: string;
  to_location?: string;
  velocity_kph?: number;
  time_delta_hours?: number;
}

export interface SecurityEvent {
  id: string;
  timestamp: string;
  user_id: string;
  resource: string;
  action: string;
  risk_score: number;
  verdict: 'ALLOW' | 'MFA' | 'DENY';
  persona: PersonaType;
}

export interface NeuralAnalysisResponse {
  verdict: 'ALLOW' | 'MFA' | 'DENY';
  risk_score: number;
  entropy: number;
  ensemble_weights: EnsembleWeights;
  deep_path: boolean;
  behavioral_analysis: BehavioralAnalysis;
  travel_anomaly?: TravelAnomaly;
  explanation: string;
  processing_stage: 'fast_path' | 'deep_path' | 'opa_enforcement';
}

export interface DemoStage {
  stage: number;
  title: string;
  description: string;
  highlight: 'persona' | 'activity' | 'neural';
}