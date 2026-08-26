ScaleSense: Configurable Roadside Multi-Radar and Camera Dataset Generation in CARLA

Abstract

Autonomous vehicles commonly rely on vehicle-mounted cameras and radars for accurate, weather-robust perception, but they are limited to the narrow viewpoints around the vehicle. But miscellaneous objects hazards could occlude the scene, preventing an autonomous system from detecting nearby road users. Roadside infrastructure addresses this limitation by examining traffic from elevated viewpoints. Yet, a single roadside sensor cannot provide extended coverage across the entire road, necessitating a network of multiple distributed radars. At present, no public datasets exist to study multi-radar camera roadside perception, limiting extensive study. To combat this issue, we developed ScaleSense, a synthetic digital twin framework for creating configurable roadside radars and camera simulations and datasets in CARLA. The current system supports configurations containing a varying number of radars, each with synchronized radar and camera capture, sensor calibration, offline data labeling, and diverse capture campaigns alongside SUMO based traffic simulations.

Project Summary

Autonomous-driving systems must detect road users in complex environments. Cameras provide visual information for object identification, but lack depth data are unreliable in poor lighting or weather. Radar provides distance and velocity measurements and is more robust in adverse environmental conditions, but observations are sparse and struggle with distinguishing object categories. Both mediums are vulnerable to occlusion.

Most autonomous-driving datasets leverage ego-vehicle perception, placing sensors directly on a vehicle, but a vehicle's sensors may miss an obscured object. Roadside sensors can provide additional perspectives by placing several radars along a roadway, facing the road. If a sensor is occluded, another sensor may still detect the hidden object. The multi-radar arrangement intends to reduce blind spots, rather than assume any individual sensor can eliminate occlusion entirely.

Existing datasets address only part of this need. RadarScenes is radar-only, while V2X-Radar provides a roadside LiDAR-camera-radar platform but uses only a single 4D radar, limiting the study of multi-radar coverage.

The research gap is not the complete absence of radar datasets, but a public dataset that combines the following properties: roadside rather than vehicle-mounted sensing; multiple radars observing overlapping parts of a road; corresponding camera observations; configurable sensor positions and orientations; controlled and repeatable traffic scenarios; and locations of ground-truth actors for labeling and evaluation.

Together, our framework facilitates synthetic data collection under conditions that would be difficult to reproduce using real roadside hardware.

Methodology

Sensor and Environment Configuration

The CARLA map can be configured with the placement and orientation of the sensors, each with editable settings. These layouts allow the same capture pipeline to be used with different levels of roadside coverage and measurements can be transformed between sensor and world coordinate systems. Sensor poses and calibration information are stored directly onto an external configuration file, rather than being hardcoded, enabling complete control of customization. Experimentation is reproducible with the stored configurations, promoting further study of multi-radar systems.

Simulation

SUMO controls the movement of vehicles and other road users, while a synchronization bridge mirrors their positions in CARLA to create configurable and repeatable traffic conditions. Random seeds and traffic parameters can be preserved so that different sensor layouts are evaluated under comparable scenarios. Once traffic is initialized, CARLA advances in a fixed simulation time step, allowing radar measurements, camera images, and actor ground truth to share consistent frame identifiers. For each frame, the framework records raw radar data, corresponding camera images, and each dynamic actor's identifier, semantic category, position, orientation, velocity, and 3D bounding box, providing synchronized ground truth for offline labeling.

Offline Radar Labeling and Post-Processing

After capture, radar detections are labeled offline to avoid slowing sensor collection. Each detection is reconstructed from its measurement and sensor pose, filtered using geometric constraints, and matched to the nearest valid actor bounding box through parallel processing for efficient labeling of large datasets. The labeled measurements are then post-processed to improve realism by refining radial velocity, modeling micro-Doppler and radar cross section, and applying an FMCW model to estimate signal quality, visibility, and measurement noise. Both the original and enhanced measurements are retained for evaluation.

Discussion

ScaleSense utilizes CARLA to create environments with various permutations of sensors along with SUMO to simulate real-world traffic scenarios. Experimenting with the application multi-radar allows for the extrapolation and optimization of roadway design and autonomous driving.

Topology-Aware Coverage Analysis

ScaleSense enables systematic study of how multi-radar topology affects coverage within a roadside region of interest. By varying the number, placement, and orientation of sensors, researchers can measure blind spots, overlapping fields of view, and the ability of multiple viewpoints to recover objects hidden by occlusion. These results can guide real-world deployments by identifying sensor arrangements that provide sufficient coverage without requiring unnecessary hardware.

Scenario-Driven Dataset Generation

The pipeline also generates labeled radar and camera data across repeatable traffic scenarios, supporting the training and evaluation of downstream perception tasks such as object detection, tracking, sensor fusion, and occupancy estimation. Because the simulated environment provides exact actor states and sensor calibration, researchers can compare models under controlled variations in traffic density, road-user composition, occlusion, and sensor placement.

Multi-View Collaborative Perception

Finally, the platform supports investigation of multi-view roadside collaborative perception. Experiments can evaluate how sensor topology influences perception accuracy, processing latency, bandwidth requirements, and robustness to individual sensor blockage or failure. These measurements can be used to determine whether a proposed deployment satisfies application quality-of-service requirements, including coverage, detection reliability, and end-to-end latency.

Contributions

The undergraduate student developed the software implementation represented in the ScaleSense repository (https://github.com/Cobby914/ConfigurableCarla), including the configurable radar-layout architecture, tested radar presets, sensor setup and spawning tools, dataset-generation pipeline, synchronized radar and camera capture, calibration and actor-state export, offline radar labeling, post-processing, campaign execution, and CARLA-SUMO integration. He is also developing the planned graphical interface for creating custom radar and camera layouts.

The other authors contributed to research discussion, evaluation, testing, and manuscript preparation. All authors reviewed the project and paper.

Conclusion

ScaleSense addresses the lack of configurable roadside datasets that combine multi-radar sensing, camera observations, controlled traffic, and simulation ground truth. By placing infrastructure-mounted radars at multiple viewpoints, the system can reduce occlusion when one sensor is blocked, while cameras provide complementary visual information and SUMO generates repeatable traffic scenarios.
