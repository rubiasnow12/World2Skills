import { useEffect, useMemo, useState } from 'react';
import type { SimulationState } from '../types';
import {
  applySimulationStep,
  createInitialSimulationState,
  simulationTasks,
} from '../utils/simulation';

const statusKeys: Array<keyof SimulationState> = [
  'object_visible',
  'object_reachable',
  'gripper_empty',
  'object_in_gripper',
  'object_in_container',
  'task_success',
];

export function ExecutionSimulation() {
  const [selectedTaskId, setSelectedTaskId] = useState(simulationTasks[1].id);
  const [state, setState] = useState<SimulationState>(() => createInitialSimulationState());
  const [activeStepIndex, setActiveStepIndex] = useState(-1);
  const [running, setRunning] = useState(false);
  const [completed, setCompleted] = useState(false);

  const selectedTask = useMemo(
    () => simulationTasks.find((task) => task.id === selectedTaskId) ?? simulationTasks[0],
    [selectedTaskId],
  );

  const activeStep = activeStepIndex >= 0 ? selectedTask.steps[activeStepIndex] : undefined;
  const stageClass = activeStep ? `stage-${activeStep.id}` : 'stage-idle';

  useEffect(() => {
    if (!running) {
      return;
    }

    if (activeStepIndex >= selectedTask.steps.length - 1) {
      setCompleted(true);
      setRunning(false);
      return;
    }

    const timer = window.setTimeout(
      () => {
        const nextIndex = activeStepIndex + 1;
        const nextStep = selectedTask.steps[nextIndex];
        setActiveStepIndex(nextIndex);
        setState((current) => applySimulationStep(current, nextStep.id));
      },
      activeStepIndex === -1 ? 250 : 850,
    );

    return () => window.clearTimeout(timer);
  }, [activeStepIndex, running, selectedTask]);

  function resetSimulation(taskId = selectedTaskId) {
    setSelectedTaskId(taskId);
    setState(createInitialSimulationState());
    setActiveStepIndex(-1);
    setRunning(false);
    setCompleted(false);
  }

  function runSimulation() {
    resetSimulation();
    setRunning(true);
  }

  return (
    <section className="section" id="execution">
      <div className="section-heading with-action">
        <div>
          <p className="eyebrow">05 / Policy / Execution Simulation</p>
          <h2>Skill-Driven Tabletop Robot Task</h2>
          <p>Procedure steps update symbolic state while the scene animates a gripper, object, and container.</p>
        </div>
        <button type="button" className="primary-button" onClick={runSimulation} disabled={running}>
          Run Simulation
        </button>
      </div>

      <div className="task-tabs" role="tablist" aria-label="Simulation tasks">
        {simulationTasks.map((task) => (
          <button
            key={task.id}
            type="button"
            className={task.id === selectedTaskId ? 'task-tab active' : 'task-tab'}
            onClick={() => resetSimulation(task.id)}
          >
            {task.name}
          </button>
        ))}
      </div>

      <div className="simulation-layout">
        <div className={`robot-stage ${stageClass}`}>
          <div className="cabinet">
            <div className={state.drawer_open ? 'drawer open' : 'drawer'}>
              <span />
            </div>
          </div>
          <div className="tabletop" />
          <div className="container-box">
            <span>container</span>
          </div>
          <div className={state.object_in_container ? 'target-object in-container' : 'target-object'} />
          <div className="gripper">
            <span />
            <span />
          </div>
          <div className="stage-label">{activeStep?.label ?? selectedTask.description}</div>
        </div>

        <div className="simulation-side">
          <article className="procedure-card">
            <h3>Procedure</h3>
            <ol>
              {selectedTask.steps.map((step, index) => (
                <li key={`${selectedTask.id}-${step.id}-${index}`} className={index === activeStepIndex ? 'active' : ''}>
                  <span>{step.label}</span>
                  <small>{step.skillId}</small>
                </li>
              ))}
            </ol>
          </article>

          <article className="status-card">
            <h3>State</h3>
            <dl>
              {statusKeys.map((key) => (
                <div key={key}>
                  <dt>{key}</dt>
                  <dd className={state[key] ? 'true' : 'false'}>{String(state[key])}</dd>
                </div>
              ))}
            </dl>
            {completed ? <p className="success-message">Task completed through Skill-based execution.</p> : null}
          </article>
        </div>
      </div>
    </section>
  );
}
