import React from 'react';
import { User, Plane, KeyRound, UserX } from 'lucide-react';
import { PersonaType } from '../types';
import { GlassCard } from './ui/GlassCard';
import { NeonButton } from './ui/NeonButton';
import { PulseIndicator } from './ui/PulseIndicator';

interface PersonaSimulatorProps {
  activePersona: PersonaType;
  onPersonaChange: (persona: PersonaType) => void;
}

const personas = [
  {
    type: 'normal' as PersonaType,
    label: 'Normal User',
    icon: User,
    color: 'green' as const,
    description: 'Standard access pattern',
  },
  {
    type: 'impossible_travel' as PersonaType,
    label: 'Impossible Travel',
    icon: Plane,
    color: 'yellow' as const,
    description: 'Geographic anomaly detected',
  },
  {
    type: 'credential_stuffing' as PersonaType,
    label: 'Credential Stuffing',
    icon: KeyRound,
    color: 'magenta' as const,
    description: 'Automated login attempts',
  },
  {
    type: 'insider_threat' as PersonaType,
    label: 'Insider Threat',
    icon: UserX,
    color: 'red' as const,
    description: 'Suspicious internal activity',
  },
];

export function PersonaSimulator({ activePersona, onPersonaChange }: PersonaSimulatorProps) {
  return (
    <div className="h-full flex flex-col">
      <GlassCard className="flex-1 flex flex-col overflow-hidden" glow="cyan">
        <div className="shrink-0 mb-4">
          <div className="flex items-center gap-3 mb-2">
            <User className="w-6 h-6 text-cyan-400" />
            <h2 className="text-xl font-bold text-cyan-300">Persona Simulator</h2>
          </div>
          <p className="text-sm text-slate-400">
            Simulate different security scenarios to test the detection system
          </p>
        </div>

        <div className="flex-1 overflow-y-auto space-y-3 min-h-0">
          {personas.map((persona) => {
            const Icon = persona.icon;
            const isActive = activePersona === persona.type;

            return (
              <div key={persona.type} className="relative">
                <NeonButton
                  variant={persona.color}
                  active={isActive}
                  onClick={() => onPersonaChange(persona.type)}
                  className="w-full text-left flex items-center gap-3"
                >
                  <Icon className="w-5 h-5" />
                  <div className="flex-1">
                    <div className="font-semibold">{persona.label}</div>
                    <div className="text-xs opacity-75">{persona.description}</div>
                  </div>
                  {isActive && <PulseIndicator color={persona.color} />}
                </NeonButton>
              </div>
            );
          })}
        </div>

        <div className="shrink-0 mt-4 pt-4 border-t border-slate-700/50">
          <div className="text-xs text-slate-500 space-y-2">
            <p className="flex items-center gap-2">
              <span className="w-2 h-2 bg-green-400 rounded-full"></span>
              Low Risk: Fast Path Processing
            </p>
            <p className="flex items-center gap-2">
              <span className="w-2 h-2 bg-yellow-400 rounded-full"></span>
              Medium Risk: Deep Path Analysis
            </p>
            <p className="flex items-center gap-2">
              <span className="w-2 h-2 bg-red-400 rounded-full"></span>
              High Risk: OPA Enforcement
            </p>
          </div>
        </div>
      </GlassCard>
    </div>
  );
}