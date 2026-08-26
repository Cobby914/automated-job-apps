EMG-Controlled Robotic Arm — In Progress

Developing an ML-driven control system that converts forearm electromyography (EMG) signals into robotic-arm movement commands.

The proposed system processes raw multichannel EMG through a pipeline resembling:
EMG acquisition → signal filtering → windowing/features → ML classification → movement command

Using Python/PyTorch for the learning pipeline and exploring simulation-based testing before full hardware deployment. Unity can provide an environment for testing gesture-to-motion mappings, controller behavior, latency, and movement accuracy before deploying commands to physical actuators.

Best statistics:
- No performance statistic yet because the project is still in development
- Future metrics should include gesture-classification F1, end-to-end command latency, number of reliably recognized gestures, and joint-position/movement error

Resume bullets:
- Developing an EMG pipeline that maps forearm muscle signals to robotic-arm commands.
- Building signal-processing and PyTorch classification workflows for EMG gesture recognition.
- Using simulation to evaluate arm-control accuracy and stability before hardware deployment.
- Designing gesture-to-motion mappings for eventual transfer to a physical robotic arm.
- Stack: Python, PyTorch, Unity, EMG.
