class Constants:
    # Capacity constraint of a rack in bps (1 Mb per ms)
    RACK_BITS_PER_MILLISEC = 1.0 * 1048576
    
    # Capacity constraint of a rack in Bps (1/8 MB per ms)
    RACK_BYTES_PER_MILLISEC = RACK_BITS_PER_MILLISEC / 8.0
    
    # Time step of Simulator (transfer 1M needs 8ms)
    SIMULATION_QUANTA = 1.0 / (RACK_BYTES_PER_MILLISEC / 1048576)
    