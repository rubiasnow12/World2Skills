import { useEffect, useState } from 'react';
import type { DataConfig, Skill } from '../types';
import { generateDataConfigs } from '../utils/generateDataConfigs';

interface DataGenerationPanelProps {
  selectedSkill: Skill;
}

export function DataGenerationPanel({ selectedSkill }: DataGenerationPanelProps) {
  const [configs, setConfigs] = useState<DataConfig[]>(() => generateDataConfigs(selectedSkill));

  useEffect(() => {
    setConfigs(generateDataConfigs(selectedSkill));
  }, [selectedSkill]);

  function handleGenerate() {
    setConfigs(generateDataConfigs(selectedSkill));
  }

  return (
    <section className="section" id="data-generation">
      <div className="section-heading with-action">
        <div>
          <p className="eyebrow">03 / Data Generation</p>
          <h2>Synthetic Training Data Configs</h2>
          <p>Hook fields on the selected Skill become scene variation knobs for training templates.</p>
        </div>
        <button type="button" className="primary-button" onClick={handleGenerate}>
          Generate Data Configs
        </button>
      </div>

      <div className="hook-strip">
        {selectedSkill.data_generation_hooks.map((hook) => (
          <span key={hook}>{hook}</span>
        ))}
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>scene_id</th>
              <th>object_shape</th>
              <th>object_pose</th>
              <th>lighting</th>
              <th>occlusion</th>
              <th>distractors</th>
              <th>surface_height</th>
              <th>expected_success_condition</th>
              <th>generated_instruction</th>
            </tr>
          </thead>
          <tbody>
            {configs.map((config) => (
              <tr key={config.scene_id}>
                <td>{config.scene_id}</td>
                <td>{config.object_shape}</td>
                <td>{config.object_pose}</td>
                <td>{config.lighting}</td>
                <td>{config.occlusion}</td>
                <td>{config.distractors}</td>
                <td>{config.surface_height}</td>
                <td>{config.expected_success_condition}</td>
                <td>{config.generated_instruction}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
