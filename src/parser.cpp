/*
parser.cpp
parse packets from a video 
*/
#include <stdio.h>
#include <stdlib.h>
extern "C"{
#include <libavcodec/avcodec.h>
}
#include <time.h>
#define INBUF_SIZE 4096

int main(int argc, char **argv){
    const char *filename, *outfilename;
    const AVCodec *codec;
    AVCodecParserContext *parser;
    AVCodecContext *c = NULL;
    FILE *f;
    uint8_t inbuf[INBUF_SIZE + AV_INPUT_BUFFER_PADDING_SIZE];
    uint8_t *data;
    size_t data_size;
    int ret;
    int eof;
    AVPacket *pkt;
    FILE *out_file;
    int pkt_count = 0;
    clock_t start_t, end_t;
    double time_cost = 0.;

    filename = argv[1];
    outfilename = argv[2];

    start_t = clock();

    out_file = fopen(outfilename, "w");
    fprintf(out_file, "pkt_size,pic_type\n");

    pkt = av_packet_alloc();
    if(!pkt){
        exit(1);
    }
    memset(inbuf+INBUF_SIZE, 0, AV_INPUT_BUFFER_PADDING_SIZE);

    codec = avcodec_find_decoder(AV_CODEC_ID_H265);
    if(!codec){
        fprintf(stderr, "Codec not found\n");
        exit(1);
    }

    parser = av_parser_init(codec->id);
    if (!parser) {
        fprintf(stderr, "parser not found\n");
        exit(1);
    }

    c = avcodec_alloc_context3(codec);
    if (!c) {
        fprintf(stderr, "Could not allocate video codec context\n");
        exit(1);
    }
    if (avcodec_open2(c, codec, NULL) < 0) {
        fprintf(stderr, "Could not open codec\n");
        exit(1);
    }

    f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Could not open %s\n", filename);
        exit(1);
    }

    do{
        data_size = fread(inbuf, 1, INBUF_SIZE, f);
        if(ferror(f)){
            break;
        }
        eof = !data_size;
        data = inbuf;
        while(data_size > 0 || eof){
            ret = av_parser_parse2(parser, c, &pkt->data, &pkt->size, data, data_size, AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
            if(ret < 0){
                fprintf(stderr, "Error while parsing\n");
                exit(1);
            }
            data += ret;
            data_size -= ret;
            if(pkt->size){
                fprintf(out_file, "%d,%d\n", pkt->size, parser->pict_type);
                pkt_count += 1;
            }
            else if(eof){
                break;
            }
        }
    } while(!eof);
    fclose(f);
    av_parser_close(parser);
    avcodec_free_context(&c);
    av_packet_free(&pkt);

    end_t = clock();
    time_cost = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    printf("%d packets parsed in %f seconds.\n", pkt_count, time_cost);

    return 0;
}