import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Định nghĩa các biến mờ
age = ctrl.Antecedent(np.arange(0, 101, 1), 'age')
cholesterol = ctrl.Antecedent(np.arange(100, 301, 1), 'cholesterol')
likelihood = ctrl.Consequent(np.arange(0, 101, 1), 'likelihood')

# Xác định hàm của biến
age['young'] = fuzz.trimf(age.universe, [0, 20, 40])
age['middle_aged'] = fuzz.trimf(age.universe, [30, 50, 70])
age['old'] = fuzz.trimf(age.universe, [60, 80, 100])

cholesterol['low'] = fuzz.trimf(cholesterol.universe, [100, 150, 200])
cholesterol['medium'] = fuzz.trimf(cholesterol.universe, [175, 225, 275])
cholesterol['high'] = fuzz.trimf(cholesterol.universe, [250, 300, 300])

likelihood['low'] = fuzz.trimf(likelihood.universe, [0, 25, 50])
likelihood['medium'] = fuzz.trimf(likelihood.universe, [40, 60, 80])
likelihood['high'] = fuzz.trimf(likelihood.universe, [70, 90, 100])

# Xác định luật mờ
rule1 = ctrl.Rule(age['young'] & cholesterol['high'], likelihood['high'])
rule2 = ctrl.Rule(age['middle_aged'] & (cholesterol['high'] | cholesterol['medium']), likelihood['medium'])
rule3 = ctrl.Rule(age['old'] & cholesterol['high'], likelihood['high'])

# Tạo hệ thống điều khiển mờ
diagnosis_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
diagnosis = ctrl.ControlSystemSimulation(diagnosis_ctrl)

# Cung cấp giá trị đầu vào
diagnosis.input['age'] = 45
diagnosis.input['cholesterol'] = 220

#Tính toán chẩn đoán
diagnosis.compute()

#In kết quả
print("Likelihood of heart disease:", diagnosis.output['likelihood'])

#Hiển thị các biến được xác định
age.view()
cholesterol.view()
likelihood.view(
