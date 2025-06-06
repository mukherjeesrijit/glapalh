from solver import solve_model, evaluate_model
from cam import viz_cam

output_model_path = rf"model.pth"
splits_file = rf"splits.csv"
cloud_file = rf"cloud.csv" 

solve_model(cloud_file, splits_file, output_model_path, num_epochs=10, batch_size=32, patience=5)
evaluate_model(output_model_path, cloud_file, splits_file, batch_size=32)

patient_id = "patientid3"
slice_num = 5
viz_cam(patient_id, slice_num, output_model_path, cloud_file, splits_file, batch_size=1)
