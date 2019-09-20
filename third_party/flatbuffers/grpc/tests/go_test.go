package testing

import (
	"../../tests/MyGame/Example"

	"context"
	"net"
	"testing"

	"google.golang.org/grpc"
)

type server struct{}

// test used to send and receive in grpc methods
var test = "Flatbuffers"
var addr = "0.0.0.0:50051"

// gRPC server store method
func (s *server) Store(context context.Context, in *Example.Monster) (*flatbuffers.Builder, error) {
	b := flatbuffers.NewBuilder(0)
	i := b.CreateString(test)
	Example.StatStart(b)
	Example.StatAddId(b, i)
	b.Finish(Example.StatEnd(b))
	return b, nil

}

// gRPC server retrieve method
func (s *server) Retrieve(context context.Context, in *Example.Stat) (*flatbuffers.Builder, error) {
	b := flatbuffers.NewBuilder(0)
	i := b.CreateString(test)
	Example.MonsterStart(b)
	Example.MonsterAddName(b, i)
	b.Finish(Example.MonsterEnd(b))
	return b, nil
}

func StoreClient(c Example.MonsterStorageClient, t *testing.T) {
	b := flatbuffers.NewBuilder(0)
	i := b.CreateString(test)
	Example.MonsterStart(b)
	Example.MonsterAddName(b, i)
	b.Finish(Example.MonsterEnd(b))
	out, err := c.Store(context.Background(), b)
	if err != nil {
		t.Fatalf("Store client failed: %v", err)
	}
	if string(out.Id()) != test {
		t.Errorf("StoreClient failed: expected=%s, got=%s\n", test, out.Id())
		t.Fail()
	}
}

func RetrieveClient(c Example.MonsterStorageClient, t *testing.T) {
	b := flatbuffers.NewBuilder(0)
	i := b.CreateString(test)
	Example.StatStart(b)
	Example.StatAddId(b, i)
	b.Finish(Example.StatEnd(b))
	out, err := c.Retrieve(context.Background(), b)
	if err != nil {
		t.Fatalf("Retrieve client failed: %v", err)
	}
	if string(out.Name()) != test {
		t.Errorf("RetrieveClient failed: expected=%s, got=%s\n", test, out.Name())
		t.Fail()
	}
}

func TestGRPC(t *testing.T) {
	lis, err := net.Listen("tcp", addr)
	if err != nil {
		t.Fatalf("Failed to listen: %v", err)
	}
	ser := grpc.NewServer(grpc.CustomCodec(flatbuffers.FlatbuffersCodec{}))
	Example.RegisterMonsterStorageServer(ser, &server{})
	go func() {
		if err := ser.Serve(lis); err != nil {
			t.Fatalf("Failed to serve: %v", err)
			t.FailNow()
		}
	}()
	conn, err := grpc.Dial(addr, grpc.WithInsecure(), grpc.WithCodec(flatbuffers.FlatbuffersCodec{}))
	if err != nil {
		t.Fatalf("Failed to connect: %v", err)
	}
	defer conn.Close()
	client := Example.NewMonsterStorageClient(conn)
	StoreClient(client, t)
	RetrieveClient(client, t)
}
