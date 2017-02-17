/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifdef __RENDER_OPENGL__
#include <GLES/gl.h>
#include <GLES/glext.h>
#endif

#include <string>
#include <map>

#include "tensorflow/examples/android/jni/object_tracking/geom.h"
#include "tensorflow/examples/android/jni/object_tracking/image-inl.h"
#include "tensorflow/examples/android/jni/object_tracking/image.h"
#include "tensorflow/examples/android/jni/object_tracking/integral_image.h"
#include "tensorflow/examples/android/jni/object_tracking/logging.h"
#include "tensorflow/examples/android/jni/object_tracking/time_log.h"
#include "tensorflow/examples/android/jni/object_tracking/utils.h"

#include "tensorflow/examples/android/jni/object_tracking/config.h"
#include "tensorflow/examples/android/jni/object_tracking/flow_cache.h"
#include "tensorflow/examples/android/jni/object_tracking/keypoint_detector.h"
#include "tensorflow/examples/android/jni/object_tracking/object_detector.h"
#include "tensorflow/examples/android/jni/object_tracking/object_tracker.h"
#include "tensorflow/examples/android/jni/object_tracking/optical_flow.h"

namespace tf_tracking {

ObjectTracker::ObjectTracker(const TrackerConfig* const config,
                             ObjectDetectorBase* const detector)
    : config_(config),
      frame_width_(config->image_size.width),
      frame_height_(config->image_size.height),
      curr_time_(0),
      num_frames_(0),
      flow_cache_(&config->flow_config),
      keypoint_detector_(&config->keypoint_detector_config),
      curr_num_frame_pairs_(0),
      first_frame_index_(0),
      frame1_(new ImageData(frame_width_, frame_height_)),
      frame2_(new ImageData(frame_width_, frame_height_)),
      detector_(detector),
      num_detected_(0) {
  for (int i = 0; i < kNumFrames; ++i) {
    frame_pairs_[i].Init(-1, -1);
  }
}


ObjectTracker::~ObjectTracker() {
  for (TrackedObjectMap::iterator iter = objects_.begin();
       iter != objects_.end(); iter++) {
    TrackedObject* object = iter->second;
    SAFE_DELETE(object);
  }
}


// Finds the correspondences for all the points in the current pair of frames.
// Stores the results in the given FramePair.
void ObjectTracker::FindCorrespondences(FramePair* const frame_pair) const {
  // Keypoints aren't found until they're found.
  memset(frame_pair->optical_flow_found_keypoint_, false,
         sizeof(*frame_pair->optical_flow_found_keypoint_) * kMaxKeypoints);
  TimeLog("Cleared old found keypoints");

  int num_keypoints_found = 0;

  // For every keypoint...
  for (int i_feat = 0; i_feat < frame_pair->number_of_keypoints_; ++i_feat) {
    Keypoint* const keypoint1 = frame_pair->frame1_keypoints_ + i_feat;
    Keypoint* const keypoint2 = frame_pair->frame2_keypoints_ + i_feat;

    if (flow_cache_.FindNewPositionOfPoint(
        keypoint1->pos_.x, keypoint1->pos_.y,
        &keypoint2->pos_.x, &keypoint2->pos_.y)) {
      frame_pair->optical_flow_found_keypoint_[i_feat] = true;
      ++num_keypoints_found;
    }
  }

  TimeLog("Found correspondences");

  LOGV("Found %d of %d keypoint correspondences",
       num_keypoints_found, frame_pair->number_of_keypoints_);
}

void ObjectTracker::NextFrame(const uint8_t* const new_frame,
                              const uint8_t* const uv_frame,
                              const int64_t timestamp,
                              const float* const alignment_matrix_2x3) {
  IncrementFrameIndex();
  LOGV("Received frame %d", num_frames_);

  FramePair* const curr_change = frame_pairs_ + GetNthIndexFromEnd(0);
  curr_change->Init(curr_time_, timestamp);

  CHECK_ALWAYS(curr_time_ < timestamp,
               "Timestamp must monotonically increase! Went from %lld to %lld"
               " on frame %d.",
               curr_time_, timestamp, num_frames_);
  curr_time_ = timestamp;

  // Swap the frames.
  frame1_.swap(frame2_);

  frame2_->SetData(new_frame, uv_frame, frame_width_, timestamp, 1);

  if (detector_.get() != NULL) {
    detector_->SetImageData(frame2_.get());
  }

  flow_cache_.NextFrame(frame2_.get(), alignment_matrix_2x3);

  if (num_frames_ == 1) {
    // This must be the first frame, so abort.
    return;
  }

  if (config_->always_track || objects_.size() > 0) {
    LOGV("Tracking %zu targets", objects_.size());
    ComputeKeypoints(true);
    TimeLog("Keypoints computed!");

    FindCorrespondences(curr_change);
    TimeLog("Flow computed!");

    TrackObjects();
  }
  TimeLog("Targets tracked!");

  if (detector_.get() != NULL && num_frames_ % kDetectEveryNFrames == 0) {
    DetectTargets();
  }
  TimeLog("Detected objects.");
}

TrackedObject* ObjectTracker::MaybeAddObject(
    const std::string& id, const Image<uint8_t>& source_image,
    const BoundingBox& bounding_box, const ObjectModelBase* object_model) {
  // Train the detector if this is a new object.
  if (objects_.find(id) != objects_.end()) {
    return objects_[id];
  }

  // Need to get a non-const version of the model, or create a new one if it
  // wasn't given.
  ObjectModelBase* model = NULL;
  if (detector_ != NULL) {
    // If a detector is registered, then this new object must have a model.
    CHECK_ALWAYS(object_model != NULL, "No model given!");
    model = detector_->CreateObjectModel(object_model->GetName());
  }
  TrackedObject* const object =
      new TrackedObject(id, source_image, bounding_box, model);

  objects_[id] = object;
  return object;
}

void ObjectTracker::RegisterNewObjectWithAppearance(
    const std::string& id, const uint8_t* const new_frame,
    const BoundingBox& bounding_box) {
  ObjectModelBase* object_model = NULL;

  Image<uint8_t> image(frame_width_, frame_height_);
  image.FromArray(new_frame, frame_width_, 1);

  if (detector_ != NULL) {
    object_model = detector_->CreateObjectModel(id);
    CHECK_ALWAYS(object_model != NULL, "Null object model!");

    const IntegralImage integral_image(image);
    object_model->TrackStep(bounding_box, image, integral_image, true);
  }

  // Create an object at this position.
  CHECK_ALWAYS(!HaveObject(id), "Already have this object!");
  if (objects_.find(id) == objects_.end()) {
    TrackedObject* const object =
        MaybeAddObject(id, image, bounding_box, object_model);
    CHECK_ALWAYS(object != NULL, "Object not created!");
  }
}

void ObjectTracker::SetPreviousPositionOfObject(const std::string& id,
                                                const BoundingBox& bounding_box,
                                                const int64_t timestamp) {
  CHECK_ALWAYS(timestamp > 0, "Timestamp too low! %lld", timestamp);
  CHECK_ALWAYS(timestamp <= curr_time_,
               "Timestamp too great! %lld vs %lld", timestamp, curr_time_);

  TrackedObject* const object = GetObject(id);

  // Track this bounding box from the past to the current time.
  const BoundingBox current_position = TrackBox(bounding_box, timestamp);

  object->UpdatePosition(current_position, curr_time_, *frame2_, false);

  VLOG(2) << "Set tracked position for " << id << " to " << bounding_box
          << std::endl;
}


void ObjectTracker::SetCurrentPositionOfObject(
    const std::string& id, const BoundingBox& bounding_box) {
  SetPreviousPositionOfObject(id, bounding_box, curr_time_);
}


void ObjectTracker::ForgetTarget(const std::string& id) {
  LOGV("Forgetting object %s", id.c_str());
  TrackedObject* const object = GetObject(id);
  delete object;
  objects_.erase(id);

  if (detector_ != NULL) {
    detector_->DeleteObjectModel(id);
  }
}

int ObjectTracker::GetKeypointsPacked(uint16_t* const out_data,
                                      const float scale) const {
  const FramePair& change = frame_pairs_[GetNthIndexFromEnd(0)];
  uint16_t* curr_data = out_data;
  int num_keypoints = 0;

  for (int i = 0; i < change.number_of_keypoints_; ++i) {
    if (change.optical_flow_found_keypoint_[i]) {
      ++num_keypoints;
      const Point2f& point1 = change.frame1_keypoints_[i].pos_;
      *curr_data++ = RealToFixed115(point1.x * scale);
      *curr_data++ = RealToFixed115(point1.y * scale);

      const Point2f& point2 = change.frame2_keypoints_[i].pos_;
      *curr_data++ = RealToFixed115(point2.x * scale);
      *curr_data++ = RealToFixed115(point2.y * scale);
    }
  }

  return num_keypoints;
}


int ObjectTracker::GetKeypoints(const bool only_found,
                                float* const out_data) const {
  int curr_keypoint = 0;
  const FramePair& change = frame_pairs_[GetNthIndexFromEnd(0)];

  for (int i = 0; i < change.number_of_keypoints_; ++i) {
    if (!only_found || change.optical_flow_found_keypoint_[i]) {
      const int base = curr_keypoint * kKeypointStep;
      out_data[base + 0] = change.frame1_keypoints_[i].pos_.x;
      out_data[base + 1] = change.frame1_keypoints_[i].pos_.y;

      out_data[base + 2] =
          change.optical_flow_found_keypoint_[i] ? 1.0f : -1.0f;
      out_data[base + 3] = change.frame2_keypoints_[i].pos_.x;
      out_data[base + 4] = change.frame2_keypoints_[i].pos_.y;

      out_data[base + 5] = change.frame1_keypoints_[i].score_;
      out_data[base + 6] = change.frame1_keypoints_[i].type_;
      ++curr_keypoint;
    }
  }

  LOGV("Got %d keypoints.", curr_keypoint);

  return curr_keypoint;
}


BoundingBox ObjectTracker::TrackBox(const BoundingBox& region,
                                    const FramePair& frame_pair) const {
  float translation_x;
  float translation_y;

  float scale_x;
  float scale_y;

  BoundingBox tracked_box(region);
  frame_pair.AdjustBox(
      tracked_box, &translation_x, &translation_y, &scale_x, &scale_y);

  tracked_box.Shift(Point2f(translation_x, translation_y));

  if (scale_x > 0 && scale_y > 0) {
    tracked_box.Scale(scale_x, scale_y);
  }
  return tracked_box;
}

BoundingBox ObjectTracker::TrackBox(const BoundingBox& region,
                                    const int64_t timestamp) const {
  CHECK_ALWAYS(timestamp > 0, "Timestamp too low! %lld", timestamp);
  CHECK_ALWAYS(timestamp <= curr_time_, "Timestamp is in the future!");

  // Anything that ended before the requested timestamp is of no concern to us.
  bool found_it = false;
  int num_frames_back = -1;
  for (int i = 0; i < curr_num_frame_pairs_; ++i) {
    const FramePair& frame_pair =
        frame_pairs_[GetNthIndexFromEnd(i)];

    if (frame_pair.end_time_ <= timestamp) {
      num_frames_back = i - 1;

      if (num_frames_back > 0) {
        LOGV("Went %d out of %d frames before finding frame. (index: %d)",
             num_frames_back, curr_num_frame_pairs_, GetNthIndexFromEnd(i));
      }

      found_it = true;
      break;
    }
  }

  if (!found_it) {
    LOGW("History did not go back far enough! %lld vs %lld",
         frame_pairs_[GetNthIndexFromEnd(0)].end_time_ -
         frame_pairs_[GetNthIndexFromStart(0)].end_time_,
         frame_pairs_[GetNthIndexFromEnd(0)].end_time_ - timestamp);
  }

  // Loop over all the frames in the queue, tracking the accumulated delta
  // of the point from frame to frame.  It's possible the point could
  // go out of frame, but keep tracking as best we can, using points near
  // the edge of the screen where it went out of bounds.
  BoundingBox tracked_box(region);
  for (int i = num_frames_back; i >= 0; --i) {
    const FramePair& frame_pair = frame_pairs_[GetNthIndexFromEnd(i)];
    SCHECK(frame_pair.end_time_ >= timestamp, "Frame timestamp was too early!");
    tracked_box = TrackBox(tracked_box, frame_pair);
  }
  return tracked_box;
}


// Converts a row-major 3x3 2d transformation matrix to a column-major 4x4
// 3d transformation matrix.
inline void Convert3x3To4x4(
    const float* const in_matrix, float* const out_matrix) {
  // X
  out_matrix[0] = in_matrix[0];
  out_matrix[1] = in_matrix[3];
  out_matrix[2] = 0.0f;
  out_matrix[3] = 0.0f;

  // Y
  out_matrix[4] = in_matrix[1];
  out_matrix[5] = in_matrix[4];
  out_matrix[6] = 0.0f;
  out_matrix[7] = 0.0f;

  // Z
  out_matrix[8] = 0.0f;
  out_matrix[9] = 0.0f;
  out_matrix[10] = 1.0f;
  out_matrix[11] = 0.0f;

  // Translation
  out_matrix[12] = in_matrix[2];
  out_matrix[13] = in_matrix[5];
  out_matrix[14] = 0.0f;
  out_matrix[15] = 1.0f;
}


void ObjectTracker::Draw(const int canvas_width, const int canvas_height,
                         const float* const frame_to_canvas) const {
#ifdef __RENDER_OPENGL__
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  glOrthof(0.0f, canvas_width, 0.0f, canvas_height, 0.0f, 1.0f);

  // To make Y go the right direction (0 at top of frame).
  glScalef(1.0f, -1.0f, 1.0f);
  glTranslatef(0.0f, -canvas_height, 0.0f);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glPushMatrix();

  // Apply the frame to canvas transformation.
  static GLfloat transformation[16];
  Convert3x3To4x4(frame_to_canvas, transformation);
  glMultMatrixf(transformation);

  // Draw tracked object bounding boxes.
  for (TrackedObjectMap::const_iterator iter = objects_.begin();
    iter != objects_.end(); ++iter) {
    TrackedObject* tracked_object = iter->second;
    tracked_object->Draw();
  }

  static const bool kRenderDebugPyramid = false;
  if (kRenderDebugPyramid) {
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    for (int i = 0; i < kNumPyramidLevels * 2; ++i) {
      Sprite(*frame1_->GetPyramidSqrt2Level(i)).Draw();
    }
  }

  static const bool kRenderDebugDerivative = false;
  if (kRenderDebugDerivative) {
    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    for (int i = 0; i < kNumPyramidLevels; ++i) {
      const Image<int32_t>& dx = *frame1_->GetSpatialX(i);
      Image<uint8_t> render_image(dx.GetWidth(), dx.GetHeight());
      for (int y = 0; y < dx.GetHeight(); ++y) {
        const int32_t* dx_ptr = dx[y];
        uint8_t* dst_ptr = render_image[y];
        for (int x = 0; x < dx.GetWidth(); ++x) {
          *dst_ptr++ = Clip(-(*dx_ptr++), 0, 255);
        }
      }

      Sprite(render_image).Draw();
    }
  }

  if (detector_ != NULL) {
    glDisable(GL_CULL_FACE);
    detector_->Draw();
  }
  glPopMatrix();
#endif
}

static void AddQuadrants(const BoundingBox& box,
                         std::vector<BoundingBox>* boxes) {
  const Point2f center = box.GetCenter();

  float x1 = box.left_;
  float x2 = center.x;
  float x3 = box.right_;

  float y1 = box.top_;
  float y2 = center.y;
  float y3 = box.bottom_;

  // Upper left.
  boxes->push_back(BoundingBox(x1, y1, x2, y2));

  // Upper right.
  boxes->push_back(BoundingBox(x2, y1, x3, y2));

  // Bottom left.
  boxes->push_back(BoundingBox(x1, y2, x2, y3));

  // Bottom right.
  boxes->push_back(BoundingBox(x2, y2, x3, y3));

  // Whole thing.
  boxes->push_back(box);
}

void ObjectTracker::ComputeKeypoints(const bool cached_ok) {
  const FramePair& prev_change = frame_pairs_[GetNthIndexFromEnd(1)];
  FramePair* const curr_change = &frame_pairs_[GetNthIndexFromEnd(0)];

  std::vector<BoundingBox> boxes;

  for (TrackedObjectMap::iterator object_iter = objects_.begin();
       object_iter != objects_.end(); ++object_iter) {
    BoundingBox box = object_iter->second->GetPosition();
    box.Scale(config_->object_box_scale_factor_for_features,
              config_->object_box_scale_factor_for_features);
    AddQuadrants(box, &boxes);
  }

  AddQuadrants(frame1_->GetImage()->GetContainingBox(), &boxes);

  keypoint_detector_.FindKeypoints(*frame1_, boxes, prev_change, curr_change);
}


// Given a vector of detections and a model, simply returns the Detection for
// that model with the highest correlation.
bool ObjectTracker::GetBestObjectForDetection(
    const Detection& detection, TrackedObject** match) const {
  TrackedObject* best_match = NULL;
  float best_overlap = -FLT_MAX;

  LOGV("Looking for matches in %zu objects!", objects_.size());
  for (TrackedObjectMap::const_iterator object_iter = objects_.begin();
      object_iter != objects_.end(); ++object_iter) {
    TrackedObject* const tracked_object = object_iter->second;

    const float overlap = tracked_object->GetPosition().PascalScore(
        detection.GetObjectBoundingBox());

    if (!detector_->AllowSpontaneousDetections() &&
        (detection.GetObjectModel() != tracked_object->GetModel())) {
      if (overlap > 0.0f) {
        return false;
      }
      continue;
    }

    const float jump_distance =
        (tracked_object->GetPosition().GetCenter() -
         detection.GetObjectBoundingBox().GetCenter()).LengthSquared();

    const float allowed_distance =
        tracked_object->GetAllowableDistanceSquared();

    LOGV("Distance: %.2f, Allowed distance %.2f, Overlap: %.2f",
         jump_distance, allowed_distance, overlap);

    // TODO(andrewharp): No need to do this verification twice, eliminate
    // one of the score checks (the other being in OnDetection).
    if (jump_distance < allowed_distance &&
        overlap > best_overlap &&
        tracked_object->GetMatchScore() + kMatchScoreBuffer <
        detection.GetMatchScore()) {
      best_match = tracked_object;
      best_overlap = overlap;
    } else if (overlap > 0.0f) {
      return false;
    }
  }

  *match = best_match;
  return true;
}


void ObjectTracker::ProcessDetections(
    std::vector<Detection>* const detections) {
  LOGV("Initial detection done, iterating over %zu detections now.",
       detections->size());

  const bool spontaneous_detections_allowed =
      detector_->AllowSpontaneousDetections();
  for (std::vector<Detection>::const_iterator it = detections->begin();
      it != detections->end(); ++it) {
    const Detection& detection = *it;
    SCHECK(frame2_->GetImage()->Contains(detection.GetObjectBoundingBox()),
          "Frame does not contain bounding box!");

    TrackedObject* best_match = NULL;

    const bool no_collisions =
        GetBestObjectForDetection(detection, &best_match);

    // Need to get a non-const version of the model, or create a new one if it
    // wasn't given.
    ObjectModelBase* model =
        const_cast<ObjectModelBase*>(detection.GetObjectModel());

    if (best_match != NULL) {
      if (model != best_match->GetModel()) {
        CHECK_ALWAYS(detector_->AllowSpontaneousDetections(),
            "Model for object changed but spontaneous detections not allowed!");
      }
      best_match->OnDetection(model,
                              detection.GetObjectBoundingBox(),
                              detection.GetMatchScore(),
                              curr_time_, *frame2_);
    } else if (no_collisions && spontaneous_detections_allowed) {
      if (detection.GetMatchScore() > kMinimumMatchScore) {
        LOGV("No match, adding it!");
        const ObjectModelBase* model = detection.GetObjectModel();
        std::ostringstream ss;
        // TODO(andrewharp): Generate this in a more general fashion.
        ss << "hand_" << num_detected_++;
        std::string object_name = ss.str();
        MaybeAddObject(object_name, *frame2_->GetImage(),
                       detection.GetObjectBoundingBox(), model);
      }
    }
  }
}


void ObjectTracker::DetectTargets() {
  // Detect all object model types that we're currently tracking.
  std::vector<const ObjectModelBase*> object_models;
  detector_->GetObjectModels(&object_models);
  if (object_models.size() == 0) {
    LOGV("No objects to search for, aborting.");
    return;
  }

  LOGV("Trying to detect %zu models", object_models.size());

  LOGV("Creating test vector!");
  std::vector<BoundingSquare> positions;

  for (TrackedObjectMap::iterator object_iter = objects_.begin();
      object_iter != objects_.end(); ++object_iter) {
    TrackedObject* const tracked_object = object_iter->second;

#if DEBUG_PREDATOR
  positions.push_back(GetCenteredSquare(
      frame2_->GetImage()->GetContainingBox(), 32.0f));
#else
    const BoundingBox& position = tracked_object->GetPosition();

    const float square_size = MAX(
        kScanMinSquareSize / (kLastKnownPositionScaleFactor *
        kLastKnownPositionScaleFactor),
        MIN(position.GetWidth(),
        position.GetHeight())) / kLastKnownPositionScaleFactor;

    FillWithSquares(frame2_->GetImage()->GetContainingBox(),
                    tracked_object->GetPosition(),
                    square_size,
                    kScanMinSquareSize,
                    kLastKnownPositionScaleFactor,
                    &positions);
  }
#endif

  LOGV("Created test vector!");

  std::vector<Detection> detections;
  LOGV("Detecting!");
  detector_->Detect(positions, &detections);
  LOGV("Found %zu detections", detections.size());

  TimeLog("Finished detection.");

  ProcessDetections(&detections);

  TimeLog("iterated over detections");

  LOGV("Done detecting!");
}


void ObjectTracker::TrackObjects() {
  // TODO(andrewharp): Correlation should be allowed to remove objects too.
  const bool automatic_removal_allowed = detector_.get() != NULL ?
      detector_->AllowSpontaneousDetections() : false;

  LOGV("Tracking %zu objects!", objects_.size());
  std::vector<std::string> dead_objects;
  for (TrackedObjectMap::iterator iter = objects_.begin();
       iter != objects_.end(); iter++) {
    TrackedObject* object = iter->second;
    const BoundingBox tracked_position = TrackBox(
        object->GetPosition(), frame_pairs_[GetNthIndexFromEnd(0)]);
    object->UpdatePosition(tracked_position, curr_time_, *frame2_, false);

    if (automatic_removal_allowed &&
        object->GetNumConsecutiveFramesBelowThreshold() >
        kMaxNumDetectionFailures * 5) {
      dead_objects.push_back(iter->first);
    }
  }

  if (detector_ != NULL && automatic_removal_allowed) {
    for (std::vector<std::string>::iterator iter = dead_objects.begin();
         iter != dead_objects.end(); iter++) {
      LOGE("Removing object! %s", iter->c_str());
      ForgetTarget(*iter);
    }
  }
  TimeLog("Tracked all objects.");

  LOGV("%zu objects tracked!", objects_.size());
}

}  // namespace tf_tracking
