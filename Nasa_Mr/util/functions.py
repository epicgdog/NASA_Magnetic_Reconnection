from datetime import datetime, date, timedelta

import numpy as np

#converts #of days into an actual date
def get_date_from_line(line):
    # Splitting the line into elements
    elements = line.strip().split(',')

    if len(elements) >= 2:
        # Assuming the first element is the year
        year = elements[0]

        # Assuming the second element is the day number
        day_num = elements[1]

        # Adjusting day num
        day_num = day_num.rjust(3 + len(day_num), '0')

        # Initializing start date
        start_date = date(int(year), 1, 1)

        # Converting to date
        res_date = start_date + timedelta(days=int(day_num) - 1)
        res = res_date.strftime("%m-%d-%Y")

        # Updating the second element in the line
        elements[1] = res

        # Joining the elements back into a string
        return ','.join(elements)
    else:
        # Return the original line if it doesn't have at least two elements
        return line
#gets date and puts in a new file
def process_file(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # Calling get_date_from_line to process each line
            processed_line = get_date_from_line(line)
            
            # Writing the modified line to the output file
            outfile.write(processed_line + '\n')

#include the first two elements of the first line of every set of 12 lines from the input file in the output file
def add_5th_element(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        lines = infile.readlines()

        for i in range(0, len(lines), 12):
            sum_of_fifth_elements = 0.0

            # Extracting the first two elements of the first line in the set
            if i < len(lines):
                first_line_elements = lines[i].strip().split(',')[:2]
                outfile.write(','.join(first_line_elements) + ',')

            for j in range(i, min(i + 12, len(lines))):
                elements = lines[j].strip().split(',')

                if len(elements) >= 5:
                    # Assuming the 5th element is a number
                    try:
                        sum_of_fifth_elements += float(elements[4])
                    except ValueError:
                        # Handle invalid values (e.g., non-numeric strings)
                        print(f"Invalid value for 5th element: {elements[4]}")

            # Rounding to the hundredth place
            sum_of_fifth_elements_rounded = round(sum_of_fifth_elements, 2)

            # Writing the rounded sum of 5th elements to the output file
            outfile.write(str(sum_of_fifth_elements_rounded) + '\n')







#switch coordinate systems
def gse_z_to_gsm(z_gse, x_gse, y_gse):
    # Calculate beta
    beta = np.arctan2(y_gse, x_gse)
    
    # Z-coordinate transformation
    z_gsm = z_gse * np.cos(beta) - x_gse * np.sin(beta)
    
    return z_gsm








