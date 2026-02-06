import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the cleaned data
df = pd.read_csv('student_data_cleaned.csv')

# Create scatter plot
plt.figure(figsize=(12, 7))

# Separate pass and fail data
pass_students = df[df['Pass'] == 1]
fail_students = df[df['Pass'] == 0]

# Plot fail students first (so pass students appear on top)
plt.scatter(fail_students['TotalStudyHours'], 
           fail_students['TotalAttendance'],
           c='#ef4444', 
           label='Fail', 
           alpha=0.6, 
           s=100,
           edgecolors='darkred',
           linewidth=0.5)

# Plot pass students
plt.scatter(pass_students['TotalStudyHours'], 
           pass_students['TotalAttendance'],
           c='#22c55e', 
           label='Pass', 
           alpha=0.6, 
           s=100,
           edgecolors='darkgreen',
           linewidth=0.5)

plt.xlabel('Total Study Hours (weekly)', fontsize=12, fontweight='bold')
plt.ylabel('Total Attendance Score', fontsize=12, fontweight='bold')
plt.title('Student Pass/Fail Distribution by Study Hours and Attendance', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('pass_fail_scatter.png', dpi=300, bbox_inches='tight')
print(f"Scatter plot saved as 'pass_fail_scatter.png'")
print(f"\nData Summary:")
print(f"Total Students: {len(df)}")
print(f"Pass Students: {len(pass_students)} ({len(pass_students)/len(df)*100:.1f}%)")
print(f"Fail Students: {len(fail_students)} ({len(fail_students)/len(df)*100:.1f}%)")

plt.show()
