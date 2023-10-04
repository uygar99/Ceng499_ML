#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <pthread.h>
#include <time.h>

#define PORT 8080
#define MAXLINE 1024

struct Message { //The box that carries 14 char and short packet number (2 bytes)
    short packet_number; //the number represents which packet is
    char message[14]; //message string of the packet
};
int n=0;
int sockfd; //file descriptor
char buffer[MAXLINE]; // buffer to keep received messages strings (useless now)
char str[MAXLINE]; // buffer to send string
struct sockaddr_in servaddr; // server address
int len=sizeof(servaddr); // length of the server address
struct Message messages[90]; // to keep all packets in a Message array
int window_size = 4; // go back n window size
int ACKs[1024] = {0}; //ACK bool array will be arranged according to received ACKs
int start; // timer start time
int stop; // timer stop time
unsigned short next_seq_num; //head for next sequence number
unsigned short send_base; //head for which message will be sent

//void *send_routine(void *p){
//    while(1){
//        send_base = 0;
//        next_seq_num = 0;
//        gets(str); // takes string
//        str[strlen(str)] = '\n'; //puts \n to the end of the string
//        int k=0; // variable to put message correct place in messages array
//        for(int i = 0; i<strlen(str); i+=14){ //loop for separating str into messages
//            struct Message message; //packet instance
//            for(int j=0;j<14;j++){
//                message.message[j] = str[j+i]; //arranging the string in the packet
//            }
//            message.packet_number=k; //arranges messages packet_number and now the packet is ready
//            messages[k]=message; // puts message in messages array
//            k++;
//        }
//        while (1){ //the loop to initilize timers to count timeouts
//            if(next_seq_num < send_base + window_size) { // sends messages if in the appropriate range
//                sendto(sockfd, (struct Message *)&messages[next_seq_num], sizeof(messages[next_seq_num]),0,(const struct sockaddr *) &servaddr,sizeof(servaddr));
//                next_seq_num++;
//            }
//            if(ACKs[k] == 1) {
//                break;
//            }
//            if(ACKs[send_base] == 1) { //if proper ACK == 1, look the time
//                send_base++;
//                if(send_base == next_seq_num){
//                    //stop_timer
//                    stop = clock() * 1000 / CLOCKS_PER_SEC; //stop the time
//                    //break;
//                }
//                else {
//                    start = clock() * 1000 / CLOCKS_PER_SEC; //start time
//                }
//            }
//            if(stop - start >= 500) { //if timeout occurs
//                start = clock() * 1000 / CLOCKS_PER_SEC; //restart the clock after timeout
//                ACKs[next_seq_num] = 0; // if timeout occurs proper ACKs will be 0
//                ACKs[next_seq_num-1] = 0;
//                ACKs[next_seq_num-2] = 0;
//                ACKs[next_seq_num-3] = 0;
//                next_seq_num = next_seq_num-4;
//            }
//        }
//    }
//    memset(&str, 0, sizeof(str));
//}

void *send_routine(void *p){
    while(1){
        send_base = 0;
        next_seq_num = 0;
        gets(str); // takes string
        str[strlen(str)] = '\n'; //puts \n to the end of the string
        int k=0; // variable to put message correct place in messages array
        for(int i = 0; i<strlen(str); i+=14){ //loop for separating str into messages
            struct Message message; //packet instance
            for(int j=0;j<14;j++){
                message.message[j] = str[j+i]; //arranging the string in the packet
            }
            message.packet_number=k; //arranges messages packet_number and now the packet is ready
            messages[k]=message; // puts message in messages array
            k++;
        }
        for(int i=0; i < k; i++) { //sends messages in appropriate order
            sendto(sockfd, (struct Message *)&messages[i], sizeof(messages[i]),0,(const struct sockaddr *) &servaddr,sizeof(servaddr));
        }
    }
    memset(&str, 0, sizeof(str));
}

//void *receive_routine(void *p){
//    while(1){
//        next_seq_num = 0; //next sequence number initilization
//        struct Message received_message; //instance of received message
//        n = recvfrom(sockfd, (struct Message *)&received_message, sizeof(received_message),0,( struct sockaddr *) &servaddr, &len);
//        if(received_message.packet_number == next_seq_num) { //check if ack (I checked using string "//a")
//            if(received_message.message[0] == '/' && received_message.message[1] == '/' && received_message.message[2] == 'a') {
//                ACKs[received_message.packet_number] = 1;
//            }
//            else { //if not ACK send ACK
//                struct Message ACK;
//                ACK.message[0] = '/';
//                ACK.message[1] = '/';
//                ACK.message[2] = 'a';
//                ACK.packet_number = next_seq_num;
//                next_seq_num++;
//                printf("%s", received_message.message);
//                sendto(sockfd, (struct Message *)&ACK, sizeof(ACK),0,(const struct sockaddr *) &servaddr,sizeof(servaddr));
//            }
//        }
//        else { //if packet is lost then send previous ACK
//            struct Message ACK;
//            ACK.message[0] = '/';
//            ACK.message[1] = '/';
//            ACK.message[2] = 'a';
//            ACK.packet_number = next_seq_num-1;
//            sendto(sockfd, (struct Message *)&ACK, sizeof(ACK),0,(const struct sockaddr *) &servaddr,sizeof(servaddr));
//        }
//        memset(&received_message, 0, sizeof(received_message)); //set received message empty
//    }
//}

void *receive_routine(void *p){
    while(1){
        next_seq_num = 0;
        struct Message received_message; //instance of received message
        n = recvfrom(sockfd, (struct Message *)&received_message, sizeof(received_message),0,( struct sockaddr *) &servaddr, &len);
        printf("%s", received_message.message);
        memset(&received_message, 0, sizeof(received_message)); //set received message empty
    }
}

int main(){
    pthread_t thread_send; //send thread
    pthread_t thread_receive; //receive thread
    if ( (sockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0 ) { //socket fd creation
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }
    memset(&servaddr, 0, sizeof(servaddr)); //making server address 0
    servaddr.sin_family = AF_INET; //IPv4
    servaddr.sin_port = htons(PORT); //PORT defined as 8080
    servaddr.sin_addr.s_addr = inet_addr("172.24.0.10"); //127.0.0.1 for local test 172.24.0.10 for vagrant
    pthread_create(&thread_send, NULL, send_routine, NULL); // send thread create and run
    pthread_create(&thread_receive, NULL, receive_routine, NULL); // receive thread create and run
    pthread_join(thread_send, NULL); //thread join
    pthread_join(thread_receive, NULL); //thread join
    close(sockfd); //close file descriptor
    return 0;
}
