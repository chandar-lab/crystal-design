#!/usr/bin/expect -f


export PATH=$PATH:/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/offline/cifs_nm/new/dir/usr/bin
# Set variables
set source_dir "/home/mila/p/prashant.govindarajan/scratch/crystal_design_project/crystal-design/crystal_design/offline/cifs_nm/new"
set destination_server "cedar.computecanada.ca"
set destination_user "pragov"
set destination_path "/home/pragov/scratch/crystal_structure_design/utils/cifs_nm"
set password "Helloworld@2022"

# Get a list of files with the .tar.gz extension in the source directory
set files [glob -nocomplain -type f "$source_dir/VAL112megnet-MG-*.tar.gz"]

# Loop through the list of files and SCP each one
for file in $files {
    # Get the base filename without the directory path
    set base_file [file tail $file]

    # SCP command for this file
    set scp_cmd "scp $file $destination_user@$destination_server:$destination_path/$base_file"

    # Spawn SCP and expect password prompt
    spawn $scp_cmd
    expect {
        "password:" {
            send "$password\r"
            exp_continue
        }
        eof
    }
}






