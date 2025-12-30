# Traffic Congestion as an Emergent Phenomenon

**Agent-Based Modeling, Simulation, and Uncertainty Analysis**

**Overview**

This project studies traffic congestion as an emergent system-level phenomenon using an agent-based model (ABM). Individual vehicles follow simple local rules (acceleration, braking, randomness), yet collectively give rise to complex global behavior such as stop-and-go waves, congestion phase transitions, and traffic jams.
The goal is to demonstrate how macroscopic traffic patterns emerge from microscopic interactions, and how uncertainty and stochasticity affect congestion risk.

**This project emphasizes:**

• Simulation-based modeling

• Emergent behavior

• Monte Carlo analysis

• Interpretability over black-box prediction

**Key Questions**

• How does traffic flow change as vehicle density increases?

• At what density does congestion emerge?

• How does randomness (driver behavior) affect stability?

• What is the distribution of outcomes under uncertainty?

**Model Summary**

• Model type: Agent-Based Model (Nagel–Schreckenberg inspired)

• Agents: Vehicles with position and velocity

• Environment: One-lane circular road

• Dynamics:

  • Acceleration up to a maximum speed
  
  • Collision avoidance
  
  • Random braking (stochasticity)
  
  • Discrete time steps

**Experiments Conducted**

1. Baseline simulation - Demonstrates emergent congestion at moderate density

2. Behavioral validation -
  • Low density → free flow
  • Medium density → congestion
  • High density → traffic jams

3. Fundamental diagram
  • Density vs traffic flow relationship
  • Identification of optimal density

4. Monte Carlo simulation -
  • Outcome distributions under stochastic driver behavior
  • Uncertainty quantification

5. Policy shock experiment - Temporary road blockage and recovery behavior

**Visual Outputs**

• Time series of average velocity

• Space–time diagrams

• Velocity distributions

• Fundamental diagram (flow vs density)

• Monte Carlo outcome distributions

