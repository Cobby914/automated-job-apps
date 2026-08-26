Unity Reinforcement Learning — Sumo / Locomotion Agent

Built and trained a physics-based reinforcement-learning agent using Unity ML-Agents, PPO, C#, and TensorBoard.

The project used curriculum learning rather than immediately attempting full sumo competition. Training progressed through tasks such as:
balance → locomotion → target navigation → block pushing → opponent/self-play

Worked with a ragdoll-like agent containing numerous independently controlled joints and a large continuous observation/action space.

A major part of the project involved diagnosing reward hacking. Earlier policies learned behaviors such as sliding, crawling, falling sideways, and other strategies that increased reward while failing to perform the intended human-like locomotion. Iteratively changed rewards and penalties based on these failures.

Best statistics:
- Controlled approximately 16 articulated joints
- Stable locomotion required 45M+ training steps
- Later self-play experimentation ran for approximately 25M steps

Resume bullets:
- Refined PPO rewards for a Unity humanoid agent controlling 16 articulated joints.
- Integrated joint rotations, velocities, targets, and arena boundaries into RL observations.
- Diagnosed crawling, sliding, and balancing reward hacks during PPO training.
- Contributed to stable locomotion after 45M+ Unity ML-Agents training steps.
- Trained self-play policies for approximately 25M additional environment steps.
- Analyzed training behavior with TensorBoard to identify reward and locomotion failures.
- Stack: Unity ML-Agents, PPO, C#, TensorBoard.
