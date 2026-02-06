import pandas as pd

# Load dataset
df = pd.read_csv("/Users/rishabhbarnwal/Documents/project/student_data.csv")

print(df.head())
print(df.info())
print(df.describe())


# Drop irrelevant columns
df.drop(columns=["StudentID", "Name"], inplace=True)

# Create target variable
df["Pass"] = (df["FinalGrade"] >= 70).astype(int)
df.drop("FinalGrade", axis=1, inplace=True)

# Handle missing values
df.fillna({
    "AttendanceRate": df["AttendanceRate"].median(),
    "StudyHoursPerWeek": df["StudyHoursPerWeek"].median(),
    "PreviousGrade": df["PreviousGrade"].median(),
    "Study Hours": df["Study Hours"].median(),
    "Attendance (%)": df["Attendance (%)"].median(),
    "Gender": df["Gender"].mode(dropna=True)[0],
    "ParentalSupport": df["ParentalSupport"].mode(dropna=True)[0],
}, inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)


gender_map = {"Female": 0, "Male": 1}
parental_support_map = {"Low": 0, "Medium": 1, "High": 2}

df["Gender"] = df["Gender"].map(gender_map)
df["ParentalSupport"] = df["ParentalSupport"].map(parental_support_map)
df["Online Classes Taken"] = (
    df["Online Classes Taken"]
    .replace({True: 1, False: 0, "True": 1, "False": 0, "Yes": 1, "No": 0})
    .fillna(0)
    .astype(int)
)

# Final NaN cleanup
numeric_cols = df.select_dtypes(include=["number"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
categorical_cols = df.select_dtypes(exclude=["number"]).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode(dropna=True)[0])

# Clamp unrealistic negatives and outliers
df["AttendanceRate"] = df["AttendanceRate"].clip(lower=0, upper=100)
df["StudyHoursPerWeek"] = df["StudyHoursPerWeek"].clip(lower=0, upper=40)
df["PreviousGrade"] = df["PreviousGrade"].clip(lower=0, upper=100)
df["Study Hours"] = df["Study Hours"].clip(lower=0, upper=10)
df["Attendance (%)"] = df["Attendance (%)"].clip(lower=0, upper=100)


# Combine similar features
df["TotalStudyHours"] = df["StudyHoursPerWeek"] + df["Study Hours"]
df["TotalAttendance"] = (df["AttendanceRate"] + df["Attendance (%)"]) / 2

# Drop old columns
df.drop(columns=[
    "StudyHoursPerWeek",
    "Study Hours",
    "AttendanceRate",
    "Attendance (%)"
], inplace=True)

from sklearn.model_selection import train_test_split

X = df.drop("Pass", axis=1)
y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save cleaned dataset for training
df.to_csv("/Users/rishabhbarnwal/Documents/project/student_data_cleaned.csv", index=False)

# Save a small preview for quick inspection
df.head(10).to_csv("/Users/rishabhbarnwal/Documents/project/student_data_head.csv", index=False)
