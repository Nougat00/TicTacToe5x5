"""
==========================================
Fuzzy Control System Documentation: Fuel Dose Calculation
==========================================

Overview
--------
This documentation provides information about a Fuzzy Control System for calculating the fuel dose
in an engine. The system is implemented using the scikit-fuzzy library in Python. It takes into account three input
variables: temperature of the coolant, demand for power, and air pressure, and provides a fuel dose as the output.

Problem Description
-------------------
The primary objective of this Fuzzy Control System is to determine the
appropriate fuel dose for an engine based on the following input variables:
- `temperature_coolant`: Temperature of the engine coolant (in degrees Celsius).
- `demand_power`: Power demand of the engine (ranging from 0 to 100).
- `air_pressure`: Air pressure in the engine (in hPa).

The output variable is:
- `fuel_dose`: The recommended fuel dose (in percentage).

Fuzzy Variables and Membership Functions
----------------------------------------
1. `temperature_coolant`:
   - Universe: 0 to 140 degrees Celsius
   - Fuzzy Sets: 'low', 'medium', 'high'

2. `demand_power`:
   - Universe: 0 to 100 percent
   - Fuzzy Sets: 'low', 'medium', 'high'

3. `air_pressure`:
   - Universe: 900 to 1100 hPa
   - Fuzzy Sets: 'low', 'medium', 'high'

4. `fuel_dose` (Consequent):
   - Universe: 0 to 100 percent
   - Fuzzy Sets: 'low', 'medium', 'high'

Rules
-----
The Fuzzy Control System defines nine rules for determining the fuel dose based on the combinations of input variables. The rules are as follows:
1. IF `temperature_coolant` is high AND `demand_power` is low AND `air_pressure` is low, THEN `fuel_dose` is low.
2. IF `temperature_coolant` is high AND `demand_power` is medium AND `air_pressure` is medium, THEN `fuel_dose` is low.
3. IF `temperature_coolant` is high AND `demand_power` is high AND `air_pressure` is high, THEN `fuel_dose` is low.
4. IF `temperature_coolant` is medium AND `demand_power` is medium AND `air_pressure` is high, THEN `fuel_dose` is medium.
5. IF `temperature_coolant` is medium AND `demand_power` is high AND `air_pressure` is high, THEN `fuel_dose` is high.
6. IF `temperature_coolant` is medium AND `demand_power` is high AND `air_pressure` is medium, THEN `fuel_dose` is medium.
7. IF `temperature_coolant` is low AND `demand_power` is low AND `air_pressure` is low, THEN `fuel_dose` is low.
8. IF `temperature_coolant` is low AND `demand_power` is high AND `air_pressure` is high, THEN `fuel_dose` is medium.
9. IF `temperature_coolant` is low AND `demand_power` is low AND `air_pressure` is medium, THEN `fuel_dose` is low.

Usage
-----
To calculate the fuel dose for a specific scenario, follow these steps:
1. Create the `fuel_ctrl` control system using the defined rules.
2. Create a `fuel_simulation` simulation object for the control system.
3. Set the input values for `temperature_coolant`, `demand_power`, and `air_pressure`.
4. Compute the output value for `fuel_dose` using the simulation.
5. Retrieve the recommended fuel dose from the `fuel_simulation` object.

Example
-------
Suppose we want to calculate the fuel dose for the following input values:
- `temperature_coolant` = 40 degrees Celsius
- `demand_power` = 100 percent
- `air_pressure` = 1100 hPa

The system recommends a fuel dose of approximately 57.9.

Created by: Jakub Gola & Bartosz Laskowski

"""

import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Tworzenie zmiennych lingwistycznych i ich funkcji przynależności
temperature_coolant = ctrl.Antecedent(np.arange(0, 141, 1), 'temperature_coolant')
demand_power = ctrl.Antecedent(np.arange(0, 101, 1), 'demand_power')
air_pressure = ctrl.Antecedent(np.arange(900, 1101, 1), 'air_pressure')
fuel_dose = ctrl.Consequent(np.arange(0, 101, 1), 'fuel_dose')

# Użycie automf do zdefiniowania funkcji przynależności
temperature_coolant.automf(3, names=['low', 'medium', 'high'])
demand_power.automf(3, names=['low', 'medium', 'high'])
air_pressure.automf(3, names=['low', 'medium', 'high'])

fuel_dose['low'] = fuzz.trimf(fuel_dose.universe, [0, 0, 50])
fuel_dose['medium'] = fuzz.trimf(fuel_dose.universe, [0, 50, 100])
fuel_dose['high'] = fuzz.trimf(fuel_dose.universe, [50, 100, 100])

# Tworzenie reguł sterowania
rule1 = ctrl.Rule(temperature_coolant['high'] & demand_power['low'] & air_pressure['low'], fuel_dose['low'])
rule2 = ctrl.Rule(temperature_coolant['high'] & demand_power['medium'] & air_pressure['medium'], fuel_dose['low'])
rule3 = ctrl.Rule(temperature_coolant['high'] & demand_power['high'] & air_pressure['high'], fuel_dose['low'])
rule4 = ctrl.Rule(temperature_coolant['medium'] & demand_power['medium'] & air_pressure['high'], fuel_dose['medium'])
rule5 = ctrl.Rule(temperature_coolant['medium'] & demand_power['high'] & air_pressure['high'], fuel_dose['high'])
rule6 = ctrl.Rule(temperature_coolant['medium'] & demand_power['high'] & air_pressure['medium'], fuel_dose['medium'])
rule7 = ctrl.Rule(temperature_coolant['low'] & demand_power['low'] & air_pressure['low'], fuel_dose['low'])
rule8 = ctrl.Rule(temperature_coolant['low'] & demand_power['high'] & air_pressure['high'], fuel_dose['medium'])
rule9 = ctrl.Rule(temperature_coolant['low'] & demand_power['low'] & air_pressure['medium'], fuel_dose['low'])

# Tworzenie systemu kontrolnego
fuel_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])

# Tworzenie symulatora
fuel_simulation = ctrl.ControlSystemSimulation(fuel_ctrl)

# Przypisanie wartości wejściowych
fuel_simulation.input['temperature_coolant'] = 80  # Temperatura cieczy chłodzącej
fuel_simulation.input['demand_power'] = 40  # Zapotrzebowanie na moc
fuel_simulation.input['air_pressure'] = 1023  # Ciśnienie powietrza

# Obliczenie wartości wyjściowej
fuel_simulation.compute()

# Wyświetlenie wyniku
print("Procent dawki paliwa: ", fuel_simulation.output['fuel_dose'])
fuel_dose.view(sim=fuel_simulation)
plt.show()
