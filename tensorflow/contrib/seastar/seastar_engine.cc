#include "tensorflow/contrib/seastar/seastar_engine.h"

#include "tensorflow/contrib/seastar/seastar_cpuset.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/env_var.h"
#include "third_party/seastar/core/alien.hh"

namespace tensorflow {

namespace {

const static int DEFAULT_CORE_NUM = 4;
const static int kWaitTimeInUs = 50000;
const static int kRetryCount = 10000;

// Seastar can't use localhost directly
string LocalhostToIp(const string& ip) {
  const auto& vec = str_util::Split(ip, ':');
  if (vec[0] == "localhost") {
    return strings::StrCat(string("127.0.0.1:"), vec[1]);
  } else {
    return ip;
  }
}

size_t GetCoreNumber() {
  int64 core_number;
  Status s = ReadInt64FromEnvVar("SEASTAR_CORE_NUMBER", DEFAULT_CORE_NUM,
                                 &core_number);
  return core_number;
}

// Force to avoid cleanup static variables and global variables
// in seastar engine which would trigger a nasty core dump
void SeastarExit(int status, void* arg) { _exit(status); }

void WaitForReady(const std::atomic_bool& a, int retry = kRetryCount) {
  while (!a.load() && retry > 0) {
    retry--;
    usleep(kWaitTimeInUs);
  }

  if (!a.load()) {
    LOG(FATAL) << "Seastar initialization failure!";
  }
}

void Server::start(uint16_t port, SeastarTagFactory* tag_factory) {
  seastar::listen_options lo;
  lo.reuse_address = true;

  listener_ = seastar::engine().listen(seastar::make_ipv4_address(port), lo);

  seastar::keep_doing([this, tag_factory] {
    return listener_->accept().then([this, tag_factory](
        seastar::connected_socket fd, seastar::socket_address addr) mutable {
      auto conn = new Connection(std::move(fd), tag_factory, addr);
      seastar::do_until([conn] { return conn->read_buf_.eof(); },
                        [conn] { return conn->Read(); })
          .then_wrapped([this, conn](auto&& f) {
            try {
              f.get();
              LOG(INFO) << "Remote close the connection:  addr = "
                        << conn->addr_;
            } catch (std::exception& ex) {
              LOG(INFO) << "Read got an exception: " << errno
                        << ", addr = " << conn->addr_;
            }
          });
    });
  }).or_terminate();
}

seastar::future<> Server::stop() { return seastar::make_ready_future<>(); }

Server::Connection::Connection(seastar::connected_socket&& fd,
                               SeastarTagFactory* tag_factory,
                               seastar::socket_address addr)
    : tag_factory_(tag_factory), addr_(addr) {
  seastar::ipv4_addr ip_addr(addr);
  struct in_addr addr_v4 {
    .s_addr = ip_addr.ip
  };
  string addr_str =
      string(inet_ntoa(addr_v4)) + ":" + std::to_string(ip_addr.port);
  channel_ = new seastar::channel(addr_str);
  fd_ = std::move(fd);
  fd_.set_nodelay(true);
  read_buf_ = fd_.input();
  channel_->init(seastar::engine().get_packet_queue(), std::move(fd_.output()));
}

Server::Connection::~Connection() { delete channel_; }

seastar::future<> Server::Connection::Read() {
  return read_buf_.read_exactly(SeastarServerTag::HEADER_SIZE)
      .then([this](auto&& header) {
        if (header.size() == 0 ||
            header.size() != SeastarServerTag::HEADER_SIZE) {
          return seastar::make_ready_future();
        }

        auto tag = tag_factory_->CreateSeastarServerTag(header, channel_);
        auto req_body_size = tag->GetRequestBodySize();
        if (req_body_size == 0) {
          tag->RecvReqDone(tensorflow::Status());
          return seastar::make_ready_future();
        }

        auto req_body_buffer = tag->GetRequestBodyBuffer();
        return read_buf_.read_exactly(req_body_size)
            .then([this, tag, req_body_size, req_body_buffer](auto&& body) {
              if (req_body_size != body.size()) {
                LOG(WARNING)
                    << "warning expected body size is:" << req_body_size
                    << ", actual body size:" << body.size();
                tag->RecvReqDone(tensorflow::Status(
                    error::UNKNOWN, "Seastar Server: read invalid msgbuf"));
                return seastar::make_ready_future<>();
              }

              memcpy(req_body_buffer, body.get(), body.size());
              tag->RecvReqDone(tensorflow::Status());
              return seastar::make_ready_future();
            });
      });
}

}  // namespace

SeastarEngine::SeastarEngine(uint16_t local,
                             SeastarWorkerService* worker_service)
    : local_(local), core_id_(0), is_server_ready_(false) {
  ::on_exit(SeastarExit, (void*)nullptr);
  tag_factory_ = new SeastarTagFactory(worker_service);
  core_number_ = GetCoreNumber();
  client_ = new SeastarClient();
  thread_ = std::thread(&SeastarEngine::AsyncStartServer, this);
}

SeastarEngine::~SeastarEngine() {
  /*
    Don't need to cleanup the variables, only one SeastarEngine
    These could be cleanup by process finished.
  */
}

seastar::channel* SeastarEngine::GetChannel(const std::string& ip) {
  WaitForReady(is_server_ready_);
  seastar::channel* ch = AsyncConnect(ip);
  WaitForReady(ch->is_init());
  return ch;
}

seastar::channel* SeastarEngine::AsyncConnect(const std::string& ip) {
  size_t core_id = core_id_++ % core_number_;
  string s = LocalhostToIp(ip);
  auto ch = new seastar::channel(s);
  alien::submit_to(core_id, [core_id, s, ch, this] {
    VLOG(2) << "client start connect core:" << core_id
            << ", connect server:" << s;
    client_->Connect(seastar::ipv4_addr{s}, s, ch, tag_factory_);
    return seastar::make_ready_future();
  });
  return ch;
}

/*
  Simple solution of get cpuset, which could be more elegant when
  a cluster scheduler, e.g. Kubernetes, provides the information
*/
void SeastarEngine::GetCpuset(char** av) {
  if (cpuset_.empty()) {
    CpusetAllocator cpuset_alloc;
    cpuset_ = cpuset_alloc.GetCpuset(core_number_);
  }
  if (cpuset_.empty()) {
    LOG(FATAL) << "Internal error when launch grpc+seastar protocol,"
               << "Please try other protocal";
  }

  *av = new char[cpuset_.size() + 1];
  memcpy(*av, cpuset_.c_str(), cpuset_.size() + 1);
}

void SeastarEngine::ConstructArgs(int* argc, char*** argv) {
  *argc = 4;

  // Set av0.
  char* av0 = new char[sizeof("useless")];
  memcpy(av0, "useless", sizeof("useless"));

  // Set av1.
  char* av1 = NULL;
  std::string str("--smp=");
  str += std::to_string(core_number_);
  av1 = new char[str.size() + 1]();
  memcpy(av1, str.c_str(), str.size());

  // Set av2.
  char* av2 = NULL;
  std::string thread_affinity("--thread-affinity=0");
  av2 = new char[thread_affinity.size() + 1]();
  memcpy(av2, thread_affinity.c_str(), thread_affinity.size());

  // Set av3 if necessary.
  char* av3 = NULL;
  av3 = new char[sizeof("--poll-mode")];
  memcpy(av3, "--poll-mode", sizeof("--poll-mode"));

  // Allocate one extra char for 'NULL' at the end.
  *argv = new char*[(*argc) + 1]();
  (*argv)[0] = av0;
  (*argv)[1] = av1;
  (*argv)[2] = av2;
  (*argv)[3] = av3;

  VLOG(2) << "Construct args result, argc: " << *(argc)
          << ", argv[0]: " << (*argv)[0] << ", argv[1]: " << (*argv)[1]
          << ", argv[2]: " << (*argv)[2] << ", argv[3]: " << (*argv)[3];
}

void SeastarEngine::AsyncStartServer() {
  int argc = 0;
  char** argv = NULL;
  ConstructArgs(&argc, &argv);

  seastar::app_template app;
  app.run_deprecated(argc, argv, [&] {
    return server_.start()
        .then([this] {
          return server_.invoke_on_all(&SeastarServer::start, local_,
                                       tag_factory_);
        })
        .then([this]() {
          is_server_ready_ = true;
          VLOG(2) << "Seastar server started successfully"
                  << ", listen port: " << local_ << ".";
          return seastar::make_ready_future();
        });
  });
}

}  // namespace tensorflow
