# Fuzzy Control System: Fuel Dose Calculation

## Overview
This documentation provides information about a Fuzzy Control System for calculating the fuel dose in an engine. The system is implemented using the scikit-fuzzy and matplotlib libs in Python. It takes into account three input variables: temperature of the coolant, demand for power, and air pressure, and provides a fuel dose as the output.

## Problem Description
The primary objective of this Fuzzy Control System is to determine the appropriate fuel dose for an engine based on the following input variables:

- `temperature_coolant`: Temperature of the engine coolant (in degrees Celsius).
- `demand_power`: Power demand of the engine (ranging from 0 to 100 percent).
- `air_pressure`: Air pressure in the engine (in hPa).

The output variable is:
- `fuel_dose`: The recommended fuel dose (in percentage).

## Fuzzy Variables and Membership Functions
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

## Rules
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

## Usage
To calculate the fuel dose for a specific scenario, follow these steps:
1. Create the `fuel_ctrl` control system using the defined rules.
2. Create a `fuel_simulation` simulation object for the control system.
3. Set the input values for `temperature_coolant`, `demand_power`, and `air_pressure`.
4. Compute the output value for `fuel_dose` using the simulation.
5. Retrieve the recommended fuel dose from the `fuel_simulation` object.

## Configuration:
1) ```git clone https://github.com/Nougat00/TicTacToe5x5.git```
2) ```pip install scikit-fuzzy```
3) ```pip install matplotlib```
4) ```python main.py```

## Example
Suppose we want to calculate the fuel dose for the following input values:
- `temperature_coolant` = 40 degrees Celsius
- `demand_power` = 100 percent
- `air_pressure` = 1100 hPa

The system recommends a fuel dose of approximately 57.9 percent.