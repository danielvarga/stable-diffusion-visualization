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


# Parameters for the filenames
first_params = [0, 1, 2]  # Can be expanded for larger grids.
second_params = [-1000, 1000]  # Can be expanded for larger grids.

first_params = [0]
import numpy as np
second_params = np.linspace(-1000, 1000, 21)

# Generate the filenames based on the given pattern.
filenames = [f"g_boost_{i}_3_{j}.png" for i in first_params for j in second_params]

# Generate the HTML content.
html_content = generate_html(filenames, len(second_params))

# This is the code that you would run to generate the HTML content.
print(html_content)
