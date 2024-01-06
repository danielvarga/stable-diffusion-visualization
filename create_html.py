import numpy as np

def header(grid_width):
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
"""
    return html_content


def footer():
    return """</body>
    </html>
"""


def grid(filenames):
    # Start the HTML string.
    html_content = """
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
    <hr/>
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


html_pieces = [header(len(shifts))]

for neuron in neurons:
    filenames = [f"{prefix}_{prompt_index}_{neuron}_{shift}.png" for prompt_index in prompt_indices for shift in shifts]
    html_pieces.append(grid(filenames))

html_pieces += [footer()]

html_content = "".join(html_pieces)

print(html_content)
