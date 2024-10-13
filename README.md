After extracting use following command to create the db in each folder:
	$> for i in $(ls); do echo "----------------------------------------------"; echo $i ; xxd -p $i; echo "END!"; done

Finally convert output to CSV
