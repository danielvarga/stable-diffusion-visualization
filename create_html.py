import numpy as np


def generate_html(filenames, grid_width):
    # Start the HTML string.
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    .grid-container {
      display: grid;
      grid-template-columns: repeat(""" + str(grid_width) + """, auto);
      padding: 5px;  /* Reduced overall container padding */
    }
    .grid-item {
      padding: 5px;  /* Reduced padding for smaller borders */
      text-align: center;
    }
    img {
      width: 300px;  /* Set a specific size for the image, adjust as needed */
      height: auto;  /* Maintain the aspect ratio */
    }
    </style>
    </head>
    <body>
    <div class="grid-container">
    """

    # Add each image to the grid.
    for filename in filenames:
        html_content += f"""
        <div class="grid-item">
            <img src="{filename}" alt="{filename}">
        </div>
        """

    # Close the HTML tags.
    html_content += """
    </div>
    </body>
    </html>
    """
    
    return html_content


prefix = "f_boost"
prompt_indices = [0, 1, 2]
neurons = [3]
shifts = [-1000, 1000]  # Can be expanded for larger grids.


prefix = "g_boost"
prompt_indices = [0]
neurons = [3]
shifts = np.linspace(-1000, 1000, 21)

prefix = "h_boost"
prompt_indices = [0,1,2,3]
neurons = [0,1,2,3]
shifts = [-1000, 1000]

prefix = "i_boost"
prompt_indices = [0,1,2]
neurons = [0,1,2,3]
shifts = np.around(np.linspace(-1000, 1000, 6)).astype(int)

prefix = "j_boost"
prompt_indices = range(10)
neurons = range(30)
shifts = np.around(np.linspace(-1000, 1000, 11)).astype(int)



# Generate the filenames based on the given pattern.
filenames = [f"{prefix}_{prompt_index}_{neuron}_{shift}.png" for neuron in neurons for prompt_index in prompt_indices for shift in shifts]

# Generate the HTML content.
html_content = generate_html(filenames, columns=len(shifts))

# This is the code that you would run to generate the HTML content.
print(html_content)
