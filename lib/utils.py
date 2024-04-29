


## IGNORE, testing gpt-generated code
@st.cache_data
def plot_wafer_map(wafer_map):
    # Identify the circle to mask out non-wafer areas
    y, x = np.ogrid[:wafer_map.shape[0], :wafer_map.shape[1]]
    center = (np.array(wafer_map.shape)-1)/2
    distance_to_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    mask = distance_to_center > center[0]

    # Apply mask
    wafer_map[mask] = 0
    
    # Create colormap
    cmap = plt.cm.viridis
    cmap.set_under(color='white')  # Set the background color

    # Define plotting
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(wafer_map, cmap=cmap, vmin=0.1)  # vmin=0.1 to use 'under' in cmap

    # Outline defects
    labeled_array, num_features = measurements.label(wafer_map==2)
    areas = measurements.sum(wafer_map==2, labeled_array, index=np.arange(labeled_array.max() + 1))
    area_map = areas[labeled_array]
    contours = np.logical_and(wafer_map == 2, area_map > 5)  # Highlight large clusters

    plt.contour(contours, colors='red', linewidths=1.5)  # Draw contour around defects clusters
    
    #plt.colorbar()
    plt.title('Wafer Map with Defective Chips Highlighted')
    plt.axis('off')  # Turn off axis numbers and ticks
    #plt.show()

    return fig