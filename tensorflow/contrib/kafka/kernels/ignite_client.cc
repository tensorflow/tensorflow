/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "ignite_client.h"

#include <sys/socket.h>
#include <arpa/inet.h>
#include <netdb.h>

#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <map>

#include "ignite_binary_object_parser.h"

namespace ignite {

// std::map<int, BinaryType*>* cache = new std::map<int, BinaryType*>();

Client::Client(std::string host, int port) {
  this->host = host;
  this->port = port;
  this->sock = -1;
}

void Client::Connect() {
  //create socket if it is not already created
  if(sock == -1) {
    //Create socket
    sock = socket(AF_INET , SOCK_STREAM , 0);
    if (sock == -1) {
      perror("Could not create socket");
    } 
  } else {   
    /* OK , nothing */  
  }     
  
  //setup address structure
  if(inet_addr(host.c_str()) == -1) {
    struct hostent *he;
    struct in_addr **addr_list;
           
    //resolve the hostname, its not an ip address
    if ( (he = gethostbyname( host.c_str() ) ) == NULL) {
      perror("Failed to resolve hostname");
      return;
    }
           
    //Cast the h_addr_list to in_addr , since h_addr_list also has the ip address in long format only
    addr_list = (struct in_addr **) he->h_addr_list;
   
    for(int i = 0; addr_list[i] != NULL; i++) {
      //strcpy(ip , inet_ntoa(*addr_list[i]) );
      server.sin_addr = *addr_list[i];
               
      break;
    }
  } else {
    server.sin_addr.s_addr = inet_addr( host.c_str() );
  }
       
  server.sin_family = AF_INET;
  server.sin_port = htons( port );
       
  //Connect to remote server
  if (connect(sock , (struct sockaddr *)&server , sizeof(server)) < 0) {
    perror("connect failed. Error");
  }
}

void Client::Disconnect() {
  
}

// int Client::JavaHashCode(std::string str) {
//   int h = 0;
//   for (char &c : str) {
//     h = 31 * h + c;
//   }
//   return h;
// }

// void Client::ScanQuery(std::string cache_name, std::vector<tensorflow::Tensor>* out_tensors) {
//   Connect(host, port);

//   // ---------- Handshake ---------- //
//   WriteInt(8);
//   WriteByte(1);
//   WriteShort(1);
//   WriteShort(0);
//   WriteShort(0);
//   WriteByte(2);

//   int handshake_res_len = ReadInt();
//   char handshake_res = ReadByte();

//   printf("Handshake result length: %d, result: %d\n", handshake_res_len, handshake_res);

//   // ---------- Scan Query ---------- //
//   WriteInt(25); // Message length
//   WriteShort(2000); // Operation code
//   WriteLong(42); // Request id
//   WriteInt(JavaHashCode(cache_name));
//   WriteByte(0); // Some flags...
//   WriteByte(101); // Filter object (NULL).
//   WriteInt(100); // Cursor page size
//   WriteInt(-1); // Partition to query
//   WriteByte(0); // Local flag

//   int res_len = ReadInt();
//   long req_id = ReadLong();
//   int status = ReadInt();
//   long cursor_id = ReadLong();
//   int row_cnt = ReadInt();
//   printf("Result length: %d\nRequest Id: %ld\nStatus: %d\nCursor : %ld\nFirst page size: %d\n", res_len, req_id, status, cursor_id, row_cnt);

//   int data_len = res_len - 8 - 4 - 8 - 4 - 1;

//   char* data = (char*) malloc(data_len);
//   recv(sock, data, data_len, 0);

//   BinaryObjectParser parser(data, out_tensors, &this);

//   for (int i = 0; i < 1; i++) {
//   	printf("------------------ Row %d -------------------\n", i);
//   	printf("-> Key ->\n");
//       parser.Parse(); // Read key
//       printf("-> Val ->\n");
//       parser.Parse(); // Read value
//   }

//   close(sock);
//   sock = -1;
// }

// BinaryType* Client::GetType(int type_id) {
//   std::map<int,BinaryType*>::iterator it = cache->find(type_id);

//   BinaryType* t;
//   if(it != cache->end()) {
//     t = it->second;
//   }
//   else {
//   	Connect(host, port);
       
//     // ---------- Handshake ---------- //
//     WriteInt(8);
//     WriteByte(1);
//     WriteShort(1);
//     WriteShort(0);
//     WriteShort(0);
//     WriteByte(2);

//     int handshake_res_len = ReadInt();
//     char handshake_res = ReadByte();

//     // ---------- Get Binary Type ----- //
//     WriteInt(14); // Message length
//     WriteShort(3002); // Operation code
//     WriteLong(49); // Request Id
//     WriteInt(type_id); // Type Id

//     int res_len = ReadInt();
//     long req_id = ReadLong();
//     int status = ReadInt();

//     char binary_exists = ReadByte();

//     BinaryType* res = new BinaryType();

//     res->type_id = ReadInt();

//     ReadByte();
//     int size = ReadInt();
//     char* type_name = (char*) malloc(size + 1);
//     type_name[size] = 0;
//     recv(sock, type_name, size, 0);

//     res->type_name = std::string(type_name);

//     char x = ReadByte();
//     if (x != 101) {
//       size = ReadInt();
//       char* affinity_key_field_name = (char*) malloc(size + 1);
//       affinity_key_field_name[size] = 0;
//       recv(sock, affinity_key_field_name, size, 0);
//       printf("x = %d, affinity_key_field_name = %s\n", x, affinity_key_field_name);
//     }

//     res->field_cnt = ReadInt();
//     res->fields = (ignite::BinaryField**) malloc(sizeof(BinaryField*) * res->field_cnt);

//     for (int i = 0; i < res->field_cnt; i++) {
//       BinaryField* field = new BinaryField();

//       char b = ReadByte();
//       int size = ReadInt();
//       char* f_name = (char*) malloc(size + 1);
//       f_name[size] = 0;
//       recv(sock, f_name, size, 0);
//       field->field_name = std::string(f_name);
//       field->type_id = ReadInt();
//       field->field_id = ReadInt();

//       res->fields[i] = field;
//     }

//     close(sock);
//     sock = -1;

//     t = res;
//     (*cache)[type_id] = t;
//   }

//   return t;
// }

char Client::ReadByte() {
  char res;
  recv(sock, &res, 1, 0);
  return res;
}

short Client::ReadShort() {
  short res;
  recv(sock, &res, 2, 0);
  return res;
}

int Client::ReadInt() {
  int res;
  recv(sock, &res, 4, 0);
  return res;
}

long Client::ReadLong() {
  long res;
  recv(sock, &res, 8, 0);
  return res;
}

void Client::ReadData(char *buf, int length) {
  recv(sock, buf, length, 0);
}

void Client::WriteByte(char data) {
  send(sock, &data, 1, 0);
}

void Client::WriteShort(short data) {
  send(sock, &data, 2, 0);
}

void Client::WriteInt(int data) {
  send(sock, &data, 4, 0);
}

void Client::WriteLong(long data) {
  send(sock, &data, 8, 0);
}

void Client::WriteData(char *buf, int length) {
  send(sock, buf, length, 0);
}

// void Client::Connect(std::string address, int port) {
//   //create socket if it is not already created
//   if(sock == -1) {
//     //Create socket
// 	  sock = socket(AF_INET , SOCK_STREAM , 0);
// 	  if (sock == -1) {
// 	    perror("Could not create socket");
// 	  } 
//   } else {   
//     /* OK , nothing */  
//   }     
	
//   //setup address structure
// 	if(inet_addr(address.c_str()) == -1) {
// 	  struct hostent *he;
// 	  struct in_addr **addr_list;
	         
// 	  //resolve the hostname, its not an ip address
// 	  if ( (he = gethostbyname( address.c_str() ) ) == NULL) {
// 	    perror("Failed to resolve hostname");
// 	    return;
// 	  }
	         
// 	  //Cast the h_addr_list to in_addr , since h_addr_list also has the ip address in long format only
// 	  addr_list = (struct in_addr **) he->h_addr_list;
	 
// 	  for(int i = 0; addr_list[i] != NULL; i++) {
// 	    //strcpy(ip , inet_ntoa(*addr_list[i]) );
// 	    server.sin_addr = *addr_list[i];
	             
// 	    break;
// 	  }
// 	} else {
// 	  server.sin_addr.s_addr = inet_addr( address.c_str() );
// 	}
	     
// 	server.sin_family = AF_INET;
// 	server.sin_port = htons( port );
	     
// 	//Connect to remote server
// 	if (connect(sock , (struct sockaddr *)&server , sizeof(server)) < 0) {
// 	  perror("connect failed. Error");
// 	}
// }

} // namespace ignite
