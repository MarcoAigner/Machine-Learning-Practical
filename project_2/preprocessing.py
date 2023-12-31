import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess(test_data: pd.DataFrame) -> pd.DataFrame:
    """ Preprocesses the data to test the exported best_model

    Renames columns to use snake_case, assigns the categorical dtypes, 
    scales numerical data and removes columns.

    Args:
        test_data (pd.DataFrame): A dataframe to evaluate the model on

    Returns:
        pd.DataFrame: Dataframe in a format accepted by the exported model
    """

    # rename columns to snake_case
    snake_case_columns = test_data.columns.map(lambda x: x.lower().replace(' ', '_').replace(
        '/', '_').replace('(', '').replace(')', '').replace('\t', '').replace('\'s', '')).to_list()
    test_data = test_data.rename(columns=dict(
        zip(test_data.columns, snake_case_columns)))  # apply the snake_case column names

    # fix typo in a column name
    test_data.rename(columns={'nacionality': 'nationality'}, inplace=True)

    # manually create a list of categorical column names
    categorical_columns = ['marital_status', 'application_mode', 'course', 'daytime_evening_attendance', 'previous_qualification', 'nationality', 'mother_qualification', 'father_qualification',
                           'mother_occupation', 'father_occupation', 'displaced', 'educational_special_needs', 'debtor', 'tuition_fees_up_to_date', 'gender', 'scholarship_holder', 'international', 'target']

    # assign the categorical dtype to respective columns
    test_data[categorical_columns] = test_data[categorical_columns].astype(
        'category')

    # numerically encode the targets
    target_mapping = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
    test_data['target'] = test_data['target'].map(arg=target_mapping)

    # standard scale numerical features
    numerical_columns = test_data.select_dtypes(include=['int64', 'float64'])
    # initialize a standard scaler with default parameters
    standard_scaler = StandardScaler()
    scaled = standard_scaler.fit_transform(
        numerical_columns)  # scale numerical columns
    test_data[numerical_columns.columns] = scaled  # override scaled columns

    # filter out columns
    columns_to_keep = ['curricular_units_2nd_sem_approved', 'curricular_units_2nd_sem_grade', 'curricular_units_1st_sem_approved', 'curricular_units_1st_sem_grade', 'admission_grade', 'curricular_units_2nd_sem_evaluations', 'tuition_fees_up_to_date', 'previous_qualification_grade', 'age_at_enrollment', 'curricular_units_1st_sem_evaluations',
                       'course', 'father_occupation', 'mother_occupation', 'gdp', 'curricular_units_2nd_sem_enrolled', 'unemployment_rate', 'father_qualification', 'mother_qualification', 'inflation_rate', 'application_mode', 'curricular_units_1st_sem_enrolled', 'scholarship_holder', 'application_order', 'debtor', 'target']

    test_data = test_data[columns_to_keep]

    return test_data
