import React, { useState, useEffect } from 'react';
import { Shield, PlayCircle, BookOpen } from 'lucide-react';
import { PersonaType, SecurityEvent, NeuralAnalysisResponse } from './types';
import { PersonaSimulator } from './components/PersonaSimulator';
import { LiveActivityPulse } from './components/LiveActivityPulse';
import { NeuralAnalysis } from './components/NeuralAnalysis';
import { TrustEntropyGauge } from './components/TrustEntropyGauge';
import { GeographicVelocity } from './components/GeographicVelocity';
import { BiometricScanAnimation } from './components/BiometricScanAnimation';
import { GuidedDemoMode } from './components/GuidedDemoMode';
import { NeonButton } from './components/ui/NeonButton';
import { analyzeSecurityEvent, createSecurityEvent } from './services/api';

function App() {
  const [activePersona, setActivePersona] = useState<PersonaType>('normal');
  const [events, setEvents] = useState<SecurityEvent[]>([]);
  const [currentAnalysis, setCurrentAnalysis] = useState<NeuralAnalysisResponse | null>(null);
  const [showMFAAnimation, setShowMFAAnimation] = useState(false);
  const [isSimulationRunning, setIsSimulationRunning] = useState(false);
  const [showGuidedDemo, setShowGuidedDemo] = useState(false);
  const [demoHighlight, setDemoHighlight] = useState<'persona' | 'activity' | 'neural' | null>(null);

  const handlePersonaChange = async (persona: PersonaType) => {
    setActivePersona(persona);

    const analysis = await analyzeSecurityEvent(persona);
    setCurrentAnalysis(analysis);

    const event = createSecurityEvent(analysis, persona, 'user_demo');
    setEvents(prev => [event, ...prev].slice(0, 10));

    if (analysis.verdict === 'MFA') {
      setShowMFAAnimation(true);
    }
  };

  const startContinuousSimulation = () => {
    setIsSimulationRunning(!isSimulationRunning);
  };

  useEffect(() => {
    if (!isSimulationRunning) return;

    const interval = setInterval(async () => {
      const personas: PersonaType[] = ['normal', 'impossible_travel', 'credential_stuffing', 'insider_threat'];
      const randomPersona = personas[Math.floor(Math.random() * personas.length)];

      const analysis = await analyzeSecurityEvent(randomPersona);
      const event = createSecurityEvent(analysis, randomPersona, `user_${Math.floor(Math.random() * 100)}`);

      setEvents(prev => [event, ...prev].slice(0, 10));
    }, 3000);

    return () => clearInterval(interval);
  }, [isSimulationRunning]);

  const getHighlightClass = (section: 'persona' | 'activity' | 'neural') => {
    if (!demoHighlight) return '';
    return demoHighlight === section
      ? 'ring-4 ring-cyan-400 ring-opacity-50 shadow-[0_0_60px_rgba(6,182,212,0.4)]'
      : 'opacity-40';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxwYXRoIGQ9Ik0zNiAxOGMzLjMxIDAgNiAyLjY5IDYgNnMtMi42OSA2LTYgNi02LTIuNjktNi02IDIuNjktNiA2LTZ6IiBzdHJva2U9IiMxZTI5M2IiIHN0cm9rZS13aWR0aD0iMC41IiBvcGFjaXR5PSIwLjMiLz48L2c+PC9zdmc+')] opacity-20"></div>

      <div className="relative z-10 container mx-auto px-6 py-8">
        <header className="mb-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="relative">
                <Shield className="w-12 h-12 text-cyan-400" />
                <div className="absolute inset-0 bg-cyan-400/20 blur-xl animate-pulse"></div>
              </div>
              <div>
                <h1 className="text-4xl font-bold bg-gradient-to-r from-cyan-300 via-blue-300 to-purple-300 bg-clip-text text-transparent">
                  Cyberpunk Security Ops
                </h1>
                <p className="text-slate-400 text-sm mt-1">
                  Real-time Threat Detection & Neural Analysis Platform
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <NeonButton
                variant="cyan"
                onClick={() => setShowGuidedDemo(true)}
                className="flex items-center gap-2"
              >
                <BookOpen className="w-4 h-4" />
                Guided Tour
              </NeonButton>
              <NeonButton
                variant={isSimulationRunning ? 'red' : 'green'}
                onClick={startContinuousSimulation}
                active={isSimulationRunning}
                className="flex items-center gap-2"
              >
                <PlayCircle className="w-4 h-4" />
                {isSimulationRunning ? 'Stop Simulation' : 'Start Simulation'}
              </NeonButton>
            </div>
          </div>
        </header>

        <div className="grid grid-cols-12 gap-6 h-[calc(100vh-180px)]">
          <div className={`col-span-3 transition-all duration-500 ${getHighlightClass('persona')}`}>
            <PersonaSimulator
              activePersona={activePersona}
              onPersonaChange={handlePersonaChange}
            />
          </div>

          <div className={`col-span-4 transition-all duration-500 ${getHighlightClass('activity')}`}>
            <LiveActivityPulse events={events} />
          </div>

          <div className="col-span-5 flex flex-col gap-6">
            <div className={`flex-1 transition-all duration-500 ${getHighlightClass('neural')}`}>
              <NeuralAnalysis analysis={currentAnalysis} />
            </div>

            <div className="grid grid-cols-2 gap-6">
              <TrustEntropyGauge entropy={currentAnalysis?.entropy || 0} />
              {currentAnalysis?.travel_anomaly && (
                <GeographicVelocity travelAnomaly={currentAnalysis.travel_anomaly} />
              )}
            </div>
          </div>
        </div>
      </div>

      <BiometricScanAnimation
        isActive={showMFAAnimation}
        onComplete={() => setShowMFAAnimation(false)}
      />

      <GuidedDemoMode
        isActive={showGuidedDemo}
        onClose={() => setShowGuidedDemo(false)}
        currentHighlight={demoHighlight}
        onHighlightChange={setDemoHighlight}
      />
    </div>
  );
}

export default App;