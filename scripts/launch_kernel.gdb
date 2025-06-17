# GDB script: Monitor the 7th parameter of hipLaunchKernel
# Usage: rocgdb -x this_script.gdb

# Set program and arguments
file /home/gns/radeon/amd-fp8-mm/build/moe_topk_checker
set args -p

# Allow pending breakpoints
set breakpoint pending on

# Disable verbose GPU wave information
set print thread-events off
set print inferior-events off

# Disable pagination to avoid pressing enter
set pagination off

# Disable verbose debug information
set debug amd-dbgapi 0

# Create log file
shell echo "=== hipLaunchKernel 7th parameter monitoring log ===" > hip_launch_log.txt
shell echo "Timestamp: $(date)" >> hip_launch_log.txt
shell echo "" >> hip_launch_log.txt

# Define breakpoint command
define log_hip_params
    # Get the 7th parameter (first parameter on stack, rsp+8)
    set $param7 = *(long*)($rsp+8)
    
    # Check if the 7th parameter is not zero
    if $param7 != 0
        printf "=== Found non-zero 7th parameter! ===\n"
        printf "7th parameter value: 0x%lx (%ld)\n", $param7, $param7
        
        # Log to file
        shell echo "Breakpoint triggered #$(date '+%H:%M:%S')" >> hip_launch_log.txt
        eval "shell echo '7th parameter: 0x%lx (%ld)' >> hip_launch_log.txt", $param7, $param7
        
        # Show all parameters for reference
        printf "All parameters:\n"
        printf "  Param1 (rdi): 0x%lx\n", $rdi
        printf "  Param2 (rsi): 0x%lx\n", $rsi  
        printf "  Param3 (rdx): 0x%lx\n", $rdx
        printf "  Param4 (rcx): 0x%lx\n", $rcx
        printf "  Param5 (r8):  0x%lx\n", $r8
        printf "  Param6 (r9):  0x%lx\n", $r9
        printf "  Param7 (stack): 0x%lx\n", $param7
        printf "  Param8 (stack): 0x%lx\n", *(long*)($rsp+16)
        
        shell echo "" >> hip_launch_log.txt
        
        # Optional: pause for user inspection (uncomment the following two lines)
        # printf "Found non-zero parameter, press enter to continue..."
        # shell read dummy
    else
        printf "7th parameter is 0, continuing execution...\n"
    end
    
    # Continue execution
    continue
end

# Set breakpoint
printf "Setting hipLaunchKernel breakpoint (pending)...\n"
break hipLaunchKernel
commands
    silent
    log_hip_params
end

# Start execution
printf "Starting hipLaunchKernel call monitoring...\n"
printf "Log will be saved to: hip_launch_log.txt\n"
printf "Press Ctrl+C to stop monitoring\n\n"

run

# Cleanup after program ends
printf "\n=== Monitoring ended ===\n"
shell echo "Monitoring end time: $(date)" >> hip_launch_log.txt
shell echo "View complete log: cat hip_launch_log.txt"