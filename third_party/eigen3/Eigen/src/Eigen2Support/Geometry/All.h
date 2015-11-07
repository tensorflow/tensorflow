#ifndef EIGEN2_GEOMETRY_MODULE_H
#define EIGEN2_GEOMETRY_MODULE_H

#include <limits>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#if EIGEN2_SUPPORT_STAGE < STAGE20_RESOLVE_API_CONFLICTS
#include "RotationBase.h"
#include "Rotation2D.h"
#include "Quaternion.h"
#include "AngleAxis.h"
#include "Transform.h"
#include "Translation.h"
#include "Scaling.h"
#include "AlignedBox.h"
#include "Hyperplane.h"
#include "ParametrizedLine.h"
#endif


#define RotationBase eigen2_RotationBase
#define Rotation2D eigen2_Rotation2D
#define Rotation2Df eigen2_Rotation2Df
#define Rotation2Dd eigen2_Rotation2Dd

#define Quaternion  eigen2_Quaternion
#define Quaternionf eigen2_Quaternionf
#define Quaterniond eigen2_Quaterniond

#define AngleAxis eigen2_AngleAxis
#define AngleAxisf eigen2_AngleAxisf
#define AngleAxisd eigen2_AngleAxisd

#define Transform   eigen2_Transform
#define Transform2f eigen2_Transform2f
#define Transform2d eigen2_Transform2d
#define Transform3f eigen2_Transform3f
#define Transform3d eigen2_Transform3d

#define Translation eigen2_Translation
#define Translation2f eigen2_Translation2f
#define Translation2d eigen2_Translation2d
#define Translation3f eigen2_Translation3f
#define Translation3d eigen2_Translation3d

#define Scaling eigen2_Scaling
#define Scaling2f eigen2_Scaling2f
#define Scaling2d eigen2_Scaling2d
#define Scaling3f eigen2_Scaling3f
#define Scaling3d eigen2_Scaling3d

#define AlignedBox eigen2_AlignedBox

#define Hyperplane eigen2_Hyperplane
#define ParametrizedLine eigen2_ParametrizedLine

#define ei_toRotationMatrix eigen2_ei_toRotationMatrix
#define ei_quaternion_assign_impl eigen2_ei_quaternion_assign_impl
#define ei_transform_product_impl eigen2_ei_transform_product_impl

#include "RotationBase.h"
#include "Rotation2D.h"
#include "Quaternion.h"
#include "AngleAxis.h"
#include "Transform.h"
#include "Translation.h"
#include "Scaling.h"
#include "AlignedBox.h"
#include "Hyperplane.h"
#include "ParametrizedLine.h"

#undef ei_toRotationMatrix
#undef ei_quaternion_assign_impl
#undef ei_transform_product_impl

#undef RotationBase
#undef Rotation2D
#undef Rotation2Df
#undef Rotation2Dd

#undef Quaternion
#undef Quaternionf
#undef Quaterniond

#undef AngleAxis
#undef AngleAxisf
#undef AngleAxisd

#undef Transform
#undef Transform2f
#undef Transform2d
#undef Transform3f
#undef Transform3d

#undef Translation
#undef Translation2f
#undef Translation2d
#undef Translation3f
#undef Translation3d

#undef Scaling
#undef Scaling2f
#undef Scaling2d
#undef Scaling3f
#undef Scaling3d

#undef AlignedBox

#undef Hyperplane
#undef ParametrizedLine

#endif // EIGEN2_GEOMETRY_MODULE_H
