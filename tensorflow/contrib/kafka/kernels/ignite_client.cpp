#include "ignite_client.h"

#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>

#include <sys/socket.h>    //socket
#include <arpa/inet.h> //inet_addr
#include <unistd.h>
#include <map>

#include "ignite_binary_object_parser.h"

namespace ignite {
	std::map<int, binary_type*>* cache = new std::map<int,binary_type*>();

	client::client() {
	    sock = -1;
	}

	int client::javaHashCode(std::string str) {
		int h = 0;
	    for (char &c : str) {
	        h = 31 * h + c;
	    }
	    return h;
	}

	void client::scan_query(std::string cache_name, std::vector<tensorflow::Tensor>* out_tensors) {
		conn("localhost", 10800);

		// ---------- Handshake ---------- //
	    write_int(8);
	    write_byte(1);
	    write_short(1);
	    write_short(0);
	    write_short(0);
	    write_byte(2);

	    int handshake_res_len = read_int();
	    char handshake_res = read_byte();

	    printf("Handshake result length: %d, result: %d\n", handshake_res_len, handshake_res);

	    // ---------- Scan Query ---------- //
	    write_int(25); // Message length
	    write_short(2000); // Operation code
	    write_long(42); // Request id
	    write_int(javaHashCode(cache_name));
	    write_byte(0); // Some flags...
	    write_byte(101); // Filter object (NULL).
	    write_int(100); // Cursor page size
	    write_int(-1); // Partition to query
	    write_byte(0); // Local flag

	    int res_len = read_int();
	    long req_id = read_long();
	    int status = read_int();
	    long cursor_id = read_long();
	    int row_cnt = read_int();
	    printf("Result length: %d\nRequest Id: %ld\nStatus: %d\nCursor : %ld\nFirst page size: %d\n", res_len, req_id, status, cursor_id, row_cnt);

	    int data_len = res_len - 8 - 4 - 8 - 4 - 1;

	    char* data = (char*) malloc(data_len);
	    recv(sock, data, data_len, 0);

	    binary_object_parser parser(data, out_tensors);

	    for (int i = 0; i < 1; i++) {
	    	printf("------------------ Row %d -------------------\n", i);
	    	printf("-> Key ->\n");
	        parser.parse(); // Read key
	        printf("-> Val ->\n");
	        parser.parse(); // Read value
	    }

	    close(sock);
	    sock = -1;
	}

	binary_type* client::get_type(int type_id) {
		std::map<int,binary_type*>::iterator it = cache->find(type_id);

		binary_type* t;
		if(it != cache->end()) {
		   	t = it->second;
		}
		else {
			conn("localhost", 10800);
     
		    // ---------- Handshake ---------- //
		    write_int(8);
		    write_byte(1);
		    write_short(1);
		    write_short(0);
		    write_short(0);
		    write_byte(2);

		    int handshake_res_len = read_int();
		    char handshake_res = read_byte();

		    // ---------- Get Binary Type ----- //
		    write_int(14); // Message length
		    write_short(3002); // Operation code
		    write_long(49); // Request Id
		    write_int(type_id); // Type Id

		    int res_len = read_int();
		    long req_id = read_long();
		    int status = read_int();

		    char binary_exists = read_byte();

		    binary_type* res = new binary_type();

		    res->type_id = read_int();

		    read_byte();
		    int size = read_int();
		    char* type_name = (char*) malloc(size + 1);
		    type_name[size] = 0;
		    recv(sock, type_name, size, 0);

		    res->type_name = std::string(type_name);

		    char x = read_byte();
		    if (x != 101) {
		    	size = read_int();
			    char* affinity_key_field_name = (char*) malloc(size + 1);
			    affinity_key_field_name[size] = 0;
			    recv(sock, affinity_key_field_name, size, 0);
			    printf("x = %d, affinity_key_field_name = %s\n", x, affinity_key_field_name);
		    }

		    res->field_cnt = read_int();
		    res->fields = (ignite::binary_field**) malloc(sizeof(binary_field*) * res->field_cnt);

		    for (int i = 0; i < res->field_cnt; i++) {
			binary_field* field = new binary_field();

		        char b = read_byte();
		        int size = read_int();
		        char* f_name = (char*) malloc(size + 1);
		        f_name[size] = 0;
		        recv(sock, f_name, size, 0);

		        field->field_name = std::string(f_name);
		        field->type_id = read_int();
		        field->field_id = read_int();

		        res->fields[i] = field;
		    }

			close(sock);
		    sock = -1;

		    t = res;
		    (*cache)[type_id] = t;
		}

		return t;
	}

	char client::read_byte() {
		char res;
    	recv(sock, &res, 1, 0);
    	return res;
	}

	short client::read_short() {
		short res;
    	recv(sock, &res, 2, 0);
    	return res;
	}

	int client::read_int() {
		int res;
    	recv(sock, &res, 4, 0);
    	return res;
	}

	long client::read_long() {
		long res;
    	recv(sock, &res, 8, 0);
    	return res;
	}

	void client::write_byte(char data) {
		send(sock, &data, 1, 0);
	}

	void client::write_short(short data) {
		send(sock, &data, 2, 0);
	}

	void client::write_int(int data) {
		send(sock, &data, 4, 0);
	}

	void client::write_long(long data) {
		send(sock, &data, 8, 0);
	}

	void client::conn(std::string address, int port) {
		//create socket if it is not already created
	    if(sock == -1)
	    {
	        //Create socket
	        sock = socket(AF_INET , SOCK_STREAM , 0);
	        if (sock == -1)
	        {
	            perror("Could not create socket");
	        }
	    }
	    else    {   /* OK , nothing */  }
	     
	    //setup address structure
	    if(inet_addr(address.c_str()) == -1)
	    {
	        struct hostent *he;
	        struct in_addr **addr_list;
	         
	        //resolve the hostname, its not an ip address
	        if ( (he = gethostbyname( address.c_str() ) ) == NULL)
	        {
	        	perror("Failed to resolve hostname");
	            return;
	        }
	         
	        //Cast the h_addr_list to in_addr , since h_addr_list also has the ip address in long format only
	        addr_list = (struct in_addr **) he->h_addr_list;
	 
	        for(int i = 0; addr_list[i] != NULL; i++)
	        {
	            //strcpy(ip , inet_ntoa(*addr_list[i]) );
	            server.sin_addr = *addr_list[i];
	             
	            break;
	        }
	    }
	     
	    //plain ip address
	    else
	    {
	        server.sin_addr.s_addr = inet_addr( address.c_str() );
	    }
	     
	    server.sin_family = AF_INET;
	    server.sin_port = htons( port );
	     
	    //Connect to remote server
	    if (connect(sock , (struct sockaddr *)&server , sizeof(server)) < 0)
	    {
	        perror("connect failed. Error");
	    }
	}
}
